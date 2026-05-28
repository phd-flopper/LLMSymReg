from pysr import PySRRegressor, TemplateExpressionSpec
import sys
import time
import os
from models.generators import PySR
from models.llm_models import LLMPlanner
from models.llm_models import LLMMutator
from models.llm_models import LLMCritic
import logging
from utils.utils import setup_logger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import omegaconf
import sympy
    
allowed_integers = [0.5, 1, 2]
julia_array_str = "[" + ", ".join([f"{float(x)}" for x in allowed_integers]) + "]"
custom_loss = f"""
function eval_loss(tree, dataset::Dataset{'{T,L}'}, options)::L where {'{T,L}'}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    mse = sum((prediction .- dataset.y).^2) / length(dataset.y)
    
    penalty = 0.0
    num_constants = 0
    
    allowed_mags = {julia_array_str}
    
    for node in tree
        if node.degree == 0 && node.constant
            num_constants += 1
            mag = abs(node.val)
            
            is_valid = any(abs(mag - allowed_val) < 1e-5 for allowed_val in allowed_mags)
            
            if !is_valid
                penalty += 100000.0
            end
        end
    end
    
    return mse + penalty
end
"""

class Trainer(object):
    def __init__(self, 
                 config:omegaconf.dictconfig.DictConfig, 
                 X:np.ndarray,
                 y:np.ndarray,
                 target:str,
                 run_id:str,
                 dom:list=None,
                 X_units:list|dict=None,
                 y_units:list|dict=None,
                 size:int=None) -> None:
        '''
        Trainer object for overall model training.
        Parameters
        ----------
        omegaconf.dictconfig.DictConfig `config` --- config file
        numpy.ndarray `X` --- data points
        numpy.ndarray `y` --- corresponding ground truth target values
        str `target` --- ground truth function (used for plotting)
        str `run_id` --- identifier of the run, used for creating directories with run logs and results
        optional list `dom` --- range of values for X
        optional list of dicts or dict `X_units` --- units of X, defaults to None i.e. dimensionless
        optional list of dicts or dict `y_units` --- units of y, defaults to None i.e. dimensionless
        '''
        self.run_dir = config['run']['run_dir'] + f'\{run_id}'
        os.makedirs(self.run_dir, exist_ok=True)
        self.pysr_dir = self.run_dir + '\pysr'
        os.makedirs(self.pysr_dir, exist_ok=True)
        self.config = config

        self.results = []
        self.logger = logging.getLogger(f'train_{run_id}')
        self.logger.setLevel(logging.INFO)
        ffh = logging.FileHandler(f"{self.run_dir}\log.txt")
        self.logger.addHandler(ffh)
        ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S'))
        
        self.base_gen_params = dict(config['generator']['base_params'])
        self.basic_params = {
            k: self.base_gen_params[k] for k in self.config['generator']['tunable_keys'] if k in self.base_gen_params}
        self.static_params = {
                                'verbosity': 0,
                                'output_directory': self.pysr_dir,
                                'model_selection': 'score',
                                'fraction_replaced_guesses': 0.95,
                                'loss_function': custom_loss,
                            }
        self.gen_config = None or {}
        self.task_constraints = None
        self.template = None or []
        self.num_vars = X.shape[1]
        self.vars_ = [f'x{i}' for i in range(self.num_vars)]
        self.X_train = X[:config['train']['size']]
        self.y_train = y[:config['train']['size']]
        self.size = size
        if self.size is not None:
            self.X_train = X[:self.size]
            self.y_train = y[:self.size]
        self.X_val = X[-config['val']['size']:]
        self.y_val = y[-config['val']['size']:]
        self.target = target
        self.dom = dom
        self.out_dom = dom
        self.X_units_info = X_units
        self.y_units_info = y_units
        self.corrupt_params = False
        self.guesses = None or []
        self.mutator_hof = []
        self.best_generator = None
        
        self.telemetry = [] 
        if self.X_units_info:
            self.X_units = [list(u.values())[0] for u in X_units]
            if all(_ is None for _ in self.X_units):
                self.X_units = None
            else:
                for i in range(len(self.X_units)):
                    if self.X_units[i] is None:
                        self.X_units[i] = ''
                    else:
                        self.X_units[i] = self.X_units[i][0]
            self.logger.info(f'Using units: {self.X_units}')
            self.results.append(f'Using units: {self.X_units}')
        else:
            self.X_units=None
        if self.y_units_info:
            self.y_units = list(y_units.values())[0]
            if self.X_units is not None:
                if self.y_units is None:
                    self.y_units = ''
                else:
                    self.logger.info(f'Using target units: {self.y_units}')
                    self.results.append(f'Using target units: {self.y_units}')
        else:
            self.y_units=None
        self.num_epochs = config['train']['num_epochs']
        self.early_stop = config['train']['early_stop']
        self.checkpoint_path = config['checkpoint_path']
        self.new_checkpoint_path = config['new_checkpoint_path'] + f'\{run_id}'
        self.plot = config['train']['plot']

        '''
        if(torch.cuda.is_available()):
            self.logger.info("CUDA is available, using GPU for computations")
        else:
            self.logger.info("CUDA is unavailable, using CPU for computations")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        '''
        
        if self.X_units is not None:
            with open(self.config['prompts']['unit_planner_user_prompt'], 'r') as f:
                if self.X_units is None:
                    self.X_units_info = None
                    self.y_units_info = None
                planner_user_prompt = f.read().format(
                    X=X, y=y, dom=dom, vars_=self.vars_, field=self.config['FIELDS'][target],
                    X_units=self.X_units_info, y_units=self.y_units_info, params=self.basic_params)
                f.close()
        else:
            with open(self.config['prompts']['planner_user_prompt'], 'r') as f:
                planner_user_prompt = f.read().format(
                    X=X, y=y, dom=dom, vars_=self.vars_, tgt=self.config['HINTS'][target],
                    params=self.basic_params)
                f.close()

        self.planner_retry = config['train']['planner_retry']
        self.planner = LLMPlanner(self.config, planner_user_prompt, self.run_dir + '\Responses\planner')
        try:
            self.gen_config, self.task_constraints, self.template = self.planner.parse_response()
            self.gen_config.update(self.update_dict(self.gen_config))
            
        except Exception as e:
            self.logger.info('Planner failed to formulate response.')
            self.results.append('Planner failed to formulate response.')
            if self.planner_retry:
                self.logger.info('Retrying...')
                self.results.append('Retrying...')
                try:
                    self.gen_config, self.task_constraints, self.template = self.planner.parse_response()
                    self.gen_config |= self.update_dict(self.gen_config)
                    
                except Exception as ee:
                    self.logger.info('Planner failed to formulate response again.')
                    self.results.append('Planner failed to formulate response again.')
                    self.logger.info(ee)
                    self.corrupt_params = True
            else:
                self.logger.info(e)
                self.results.append(e)
                self.corrupt_params = True
        else:
            self.logger.info('Planner successfully formulated constraints:')
            self.logger.info(f'Proposed generator configurations:\n{self.gen_config}')
            self.logger.info(f'Proposed task constraints:\n{self.task_constraints}')
            self.logger.info(f'Proposed template:\n{self.template}')

        
        try:
            self.generator = PySRRegressor(guesses=self.template, niterations=0,
                                           **(self.static_params | self.gen_config))
            self.generator.fit(self.X_train, self.y_train, X_units=self.X_units, y_units=self.y_units)
            self.guesses  = self.template
        except Exception as e:
            self.logger.info('Planner failed to formulate valid template. Initiating with defaults.')
            self.results.append('Planner failed to formulate valid template. Initiating with defaults.')
            self.corrupt_params = True
        
        if self.corrupt_params == True:
            self.generator = PySRRegressor(niterations = 1 * self.config['train']['factor'],
                **(self.base_gen_params | self.static_params))
            self.dynamic_params = self.base_gen_params
        
        else:
            self.generator = PySRRegressor(niterations = 1 * self.config['train']['factor'], guesses=self.guesses,
                **(self.gen_config | self.static_params))
            self.dynamic_params = self.gen_config
        
        self.mutator = LLMMutator(self.config, self.task_constraints, self.run_dir + '\Responses\mutator')
        self.critic = LLMCritic(self.config, self.task_constraints, self.run_dir + '\Responses\critic')
        self.critic_response = None or {}
        self.mutator_response = []
        self.eqs = ['x0']
        self.ideas = None
        self.cur_epoch = 0
        self.start = 1

        if config['checkpoint_path'] is not None:
            self.logger.info(f"Continue training from checkpoint: {self.checkpoint_path}")
            self.generator = PySRRegressor.from_file(run_directory=self.checkpoint_path)
        else:
            self.logger.info("Initiating new training run")  
            
    def train(self, epoch:int) -> float:
        self.logger.info(f'Initiating train run, epoch {epoch}.')
        start_time = time.time()
        epoch_loss = np.inf
        operators = {
                        'binary_operators': self.dynamic_params['binary_operators'],
                        'unary_operators': self.dynamic_params['unary_operators']
                    }
        if epoch > 1:
            try:
                critic_response, ideas = self.critic.parse_response(self.eqs, 
                                                                    self.dynamic_params, 
                                                                    self.telemetry, 
                                                                    self.vars_,
                                                                    self.template,
                                                                    self.basic_params)
                critic_response |= self.update_dict(critic_response)
            except Exception as e:
                self.logger.info('Critic failed to propose trajectory.')
                self.logger.info(e)
                self.results.append(f'Critic failed to propose trajectory on epoch: {self.cur_epoch}.')
                pass
            else:
                self.critic_response = critic_response
                self.ideas = ideas
                self.logger.info('Critic successfully proposed trajectory.')
                self.logger.info(f'Proposed generator configurations:\n{self.critic_response}')

            try:
                mutator_response = self.mutator.parse_response(self.eqs, 
                                                               self.vars_, 
                                                               self.ideas, 
                                                               operators,
                                                               self.template)
            except Exception as e:
                self.logger.info('Mutator failed to propose equations.')
                self.logger.info(e)
                self.results.append(f'Mutator failed to propose equations on epoch: {self.cur_epoch}.')
                pass
            else:
                self.mutator_response = mutator_response
                self.logger.info('Mutator successfully proposed equations.')
                self.logger.info(f'Proposed equations: \n{mutator_response}')

            #critic_corrupt = False
            try:
                tmp_mod = PySRRegressor(niterations=0, verbosity=0)
                tmp_mod.set_params(**(self.dynamic_params | self.static_params | self.critic_response))
                tmp_mod.set_params(output_directory=self.pysr_dir + '\TEMP')
                tmp_mod.fit(self.X_train, self.y_train, X_units=self.X_units, y_units=self.y_units)
            except Exception as e:
                self.logger.info(f'Critic corrupt on epoch {epoch}.')
                self.logger.info(e)
                self.results.append(f'Critic corrupt on epoch {epoch}.')
                #critic_corrupt = True
                pass
            else:
                self.dynamic_params |= self.critic_response
            finally:
                del tmp_mod
                
            #mutator_corrupt = False
            try:
                tmp_mod1 = PySRRegressor(niterations=1, verbosity=0)
                template = None or []
                if self.corrupt_params == False:
                    template = self.template
                tmp_mod1.set_params(guesses=self.mutator_response + template, 
                                   **(self.dynamic_params | self.static_params))
                tmp_mod1.set_params(output_directory=self.pysr_dir + '\TEMP')
                tmp_mod1.fit(self.X_train, self.y_train, X_units=self.X_units, y_units=self.y_units)
            except Exception as e:
                self.logger.info(f'Mutator corrupt on epoch {epoch}.')
                self.logger.info(e)
                self.results.append(f'Mutator corrupt on epoch {epoch}.')
                #mutator_corrupt = True
                pass
            else:
                self.guesses = template + self.mutator_response
            finally:
                del tmp_mod1
            
            self.generator = PySRRegressor(niterations = 1 * self.config['train']['factor'], 
                                           guesses=self.guesses,
                                           **(self.dynamic_params | self.static_params))
            
        try:
            self.generator.fit(self.X_train, self.y_train, X_units=self.X_units, y_units=self.y_units)
        except Exception as e:
            self.logger.info('Something went wrong.')
            self.results.append(f'Unable to fit on epoch: {self.cur_epoch}.')
            self.logger.info(e)
            return epoch_loss
        
        self.eqs = self.generator.equations_['sympy_format'].to_list()
        y_pred = self.generator.predict(self.X_train)
        epoch_loss = MSE(y_pred, self.y_train)
        
        end_time = time.time()
        self.logger.info('')
        self.logger.info(f'Finished training on epoch {epoch} in {end_time - start_time} seconds')
        self.logger.info(f'Training epoch {epoch} loss: {epoch_loss}')
        return epoch_loss

    def validation(self, epoch:int) -> list:
        epoch_loss = 0.0
        score = 0
        self.logger.info(f'Initiating validation run, epoch {epoch}.')
        start_time = time.time()
        y_pred = self.generator.predict(self.X_val)
        epoch_loss = MSE(y_pred, self.y_val)
        score = r2_score(self.y_val, y_pred)
        mae_loss = MAE(y_pred, self.y_val)
        mutator_output = {'equation': None, 'score': None}
        if epoch > 1:
            for func in self.mutator_response:
                try: 
                    func = func.replace("^", "**")
                    X_val = [self.X_val[:, i] for i in range(self.X_val.shape[1])]
                    function = sympy.lambdify(self.vars_, func)
                    function_score = r2_score(self.y_val, function(*X_val))
                    if mutator_output['score'] is not None:
                        if function_score > mutator_output['score']:
                            mutator_output['score'] = function_score
                            mutator_output['equation'] = func
                    else:
                        mutator_output['score'] = function_score
                        mutator_output['equation'] = func
                except:
                    continue
        
        self.telemetry.append({f'Epoch {epoch} $R^2$ score':score, 
                               f'Epoch {epoch} MSE':epoch_loss,
                               f'Epoch {epoch} MAE':mae_loss,
                              f'Epoch {epoch} Mutator proposal': mutator_output})
        
        self.mutator_hof.append(mutator_output)
        end_time = time.time()
        self.logger.info('')
        self.logger.info(f'Finished validation on epoch {epoch} in {end_time - start_time} seconds')
        self.logger.info(f'Validation epoch {epoch} loss: {epoch_loss}')
        self.logger.info(f'Validation epoch {epoch} R^2 score: {score}')
        if mutator_output['score'] is not None:
            score = max(score, mutator_output['score'])
        return epoch_loss, score

    def run(self) -> list:
        if self.num_epochs == 0:
            return self.results
        self.marker = True
        train_loss = []
        val_loss = []
        scores = []
        v_loss = np.inf
        best_loss = v_loss
        no_improve = 0
        loss_fname = self.run_dir + '\losses.txt'
        while self.cur_epoch < self.num_epochs:
            if no_improve >= self.early_stop:
                self.logger.info(f'No improvement for {no_improve} consecutive epochs. Terminating run.')
                self.results.append(
                    f'No improvement for {no_improve} consecutive epochs. Run terminated on epoch:{self.cur_epoch}.')
                self.marker = False
                break
            start_time = time.time()
            self.cur_epoch += 1
            self.logger.info(f'Initiating epoch {self.cur_epoch}. Loss {v_loss}')
            self.logger.info('')
            t_loss = self.train(self.cur_epoch)
            if t_loss == np.inf:
                self.cur_epoch -= 1
                self.marker = False
                break
            v_loss, score = self.validation(self.cur_epoch)

            f = open(loss_fname, "a")
            f.write(f'Epoch: {self.cur_epoch}, train_loss: {t_loss}, val_loss: {v_loss}\n')
            f.close()

            train_loss.append(t_loss)
            val_loss.append(v_loss)
            scores.append(score)

            if v_loss >= best_loss:
                no_improve += 1
                self.logger.info(
                    f'Could not find best model for {no_improve} consecutive epochs. Current best loss {best_loss}.')

            else:
                best_loss = v_loss
                no_improve = 0
                self.logger.info(
                    f'New best model found. New best loss {best_loss}.')
                #self.save_checkpoint(self.cur_epoch, best=True)
                self.best_generator = self.generator
                self.logger.info('Saving best model checkpoint.')
            if score >= 0.999:
                self.logger.info(f'Stopping critetion met. R^2 score {score}.')
                self.results.append(f'Stopping critetion met on epoch {self.cur_epoch}. R^2 score {score}.')
                self.marker = False
                break
            end_time = time.time()
            self.logger.info(f'Finished epoch {self.cur_epoch}, in {end_time - start_time} seconds.')
        if self.marker:
            self.logger.info(f'Finished training for {self.cur_epoch} epochs.')
            self.results.append(f'Finished full training run for {self.cur_epoch} epochs.')
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.scores = scores
        
        if self.plot is not None:
            os.makedirs(self.run_dir + '\plots', exist_ok=True)
            self._plot_losses()
            self._plot_scores()
            self._plot_function()

        self._save_latex()
        self._save_hof()
        
        return self.results
    
    def _plot_losses(self) -> None:
        self.logger.info('Plotting losses...')
        plt.title("Loss of train and val")
        x = [i for i in range(self.start, self.cur_epoch + 1)]
        plt.plot(x, self.train_loss, 'b-', label=u'train_loss', linewidth=0.8)
        plt.plot(x, self.val_loss, 'c-', label=u'val_loss', linewidth=0.8)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        fname = self.run_dir + '\plots\loss'
        self.logger.info(f'Plotted image saved at {fname}')
        plt.savefig(fname)
        plt.savefig(fname + '.svg')
        plt.show()
        return

    def _plot_scores(self) -> None:
        self.logger.info('Plotting scores...')
        plt.title("Score on val")
        x = [i for i in range(self.start, self.cur_epoch + 1)]
        plt.plot(x, self.scores, 'r-', linewidth=0.8)
        plt.ylabel('score')
        plt.xlabel('epoch')
        fname = self.run_dir + '\plots\score'
        self.logger.info(f'Plotted image saved at {fname}')
        plt.savefig(fname)
        plt.savefig(fname + '.svg')
        plt.show()
        return
    
    def _plot_function(self) -> None:
        if self.num_vars > 1:
            self.logger.info(f'{self.target} has too many variables.')
            return
        if self.target not in self.config['1d_funcs'].keys():
            self.logger.info(f'Unable to plot {self.target}.')
            return
        self.logger.info('Plotting functions...')
        tgt = eval(self.config['1d_funcs'][self.target])
        dom = np.linspace(self.dom[0], self.dom[1], self.config['train']['size'])
        if self.size is not None:
            dom = np.linspace(self.dom[0], self.dom[1], self.size)
        y_pred = self.best_generator.predict(dom.reshape(-1, 1))
        y_true = tgt(dom)
        plt.plot(dom, y_true, 'r', alpha=0.5, label='Ground truth')
        plt.plot(dom, y_pred, 'c--', label='Prediction')
        plt.legend()
        plt.title("Plots of predicted and target functions")
        plt.xlabel("x")
        plt.ylabel(self.target)
        plt.grid(True)
        fname = self.run_dir + '\plots\Func'
        self.logger.info(f'Plotted image saved at {fname}')
        plt.savefig(fname)
        plt.savefig(fname + '.svg')
        plt.show()
        tgt = eval(self.config['1d_funcs'][self.target])
        dom = np.linspace(self.dom[0] * 10, self.dom[1] * 10, self.config['train']['size'])
        if self.size is not None:
            dom = np.linspace(self.dom[0] * 10, self.dom[1] * 10, self.size)
        y_pred = self.best_generator.predict(dom.reshape(-1, 1))
        y_true = tgt(dom)
        plt.plot(dom, y_true, 'r', alpha=0.5, label='Ground truth')
        plt.plot(dom, y_pred, 'c--', label='Prediction')
        plt.legend()
        plt.title("Plots of predicted and target functions (enhanced domain)")
        plt.xlabel("x")
        plt.ylabel(self.target)
        plt.grid(True)
        fname = self.run_dir + '\plots\BiggerFunc'
        self.logger.info(f'Plotted image saved at {fname}')
        plt.savefig(fname)
        plt.savefig(fname + '.svg')
        plt.show()
        return
    
    def _save_latex(self) -> None:
        try:
            with open(self.run_dir + '\LaTeX.txt', 'w+') as f:
                f.write(self.best_generator.latex_table())
                f.close()
        except:
            return
        return
    
    def _save_hof(self) -> None:
        with open(self.run_dir + '\mutator_HoF.txt', 'a+') as f:
            for eq in self.mutator_hof:
                f.write(f'{eq["equation"]}, R^2 score: {eq["score"]} \n')
            f.write('\n\n')
            f.close()
        return
    
    def get_best_mutator_equation(self) -> list:
        equation = ''
        score = -np.inf
        for pair in self.mutator_hof:
            if pair['score'] > score:
                equation = pair['equation']
                score = pair['score']
        return [equation, score]
    
    def update_dict(self, dicta:dict) -> dict:
        res = {'constraints': dict(), 'nested_constraints': dict()}
        if 'log' in dicta['unary_operators']:
            res['constraints']['log'] = self.base_gen_params['constraints']['log']
            if 'exp' in dicta['unary_operators']:
                res['nested_constraints']['log'] = self.base_gen_params['nested_constraints']['log']
            else:
                res['nested_constraints']['log'] = {'log': 0}
        if 'cos' in dicta['unary_operators']:
            res['constraints']['cos'] = self.base_gen_params['constraints']['cos']
            if 'sin' in dicta['unary_operators']:
                res['nested_constraints']['cos'] = self.base_gen_params['nested_constraints']['cos']
            else:
                res['nested_constraints']['cos'] = {'cos': 0}
        if 'sin' in dicta['unary_operators']:
            res['constraints']['sin'] = self.base_gen_params['constraints']['sin']
            if 'cos' in dicta['unary_operators']:
                res['nested_constraints']['sin'] = self.base_gen_params['nested_constraints']['sin']
            else:
                res['nested_constraints']['sin'] = {'sin': 0}
        if 'exp' in dicta['unary_operators']:
            if 'log' in dicta['unary_operators']:
                res['nested_constraints']['exp'] = self.base_gen_params['nested_constraints']['exp']
            else:
                res['nested_constraints']['exp'] = {'exp': 0}
        if 'sqrt' in dicta['unary_operators']:
            res['constraints']['sqrt'] = self.base_gen_params['constraints']['sqrt']
        if '^' in dicta['binary_operators']:
            res['constraints']['pow'] = self.base_gen_params['constraints']['pow']
        return res
            
        
    def save_checkpoint(self, epoch:int, best:bool=True):
        pass