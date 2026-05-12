import matplotlib.pyplot as plt
import numpy as np

def _plot_recovery_rates(ng_20_name:str, ng_100_name:str, out_fname:str, g_title:str) -> None:
    c_ng_20 = np.load(ng_20_name)
    c_ng_100 = np.load(ng_100_name)
    ng_20 = [i * 10 for i in c_ng_20]
    ng_100 = [i * 10 for i in c_ng_100]
    
    eq = np.arange(1, 13)
    plt.xticks(eq) 
    width = 0.2
    
    plt.bar(eq - width, ng_20, width, label='Sample size: 20', color='green', linestyle='-')
    plt.bar(eq, ng_100, width, label='Sample size: 100', color='orange', linestyle='-')
    
    plt.title(g_title)
    plt.xlabel('Nguyen equation number')
    plt.ylabel('Recovery rate')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(f'{out_fname}.svg')
    plt.savefig(f'{out_fname}')
    plt.show()
    return

def _plot_ablation() -> None:
    b_ng_20 = np.load('recovery_rates/b_ng_20.npy') #baseline
    b_ng_100 = np.load('recovery_rates/b_ng_100.npy') #baseline
    
    c_ng_20 = np.load('recovery_rates/c_ng_20.npy') #no_critic
    c_ng_100 = np.load('recovery_rates/c_ng_100.npy') #no_critic
    
    p_ng_20 = np.load('recovery_rates/p_ng_20.npy') #no_planner
    p_ng_100 = np.load('recovery_rates/p_ng_100.npy') #no_planner
    
    m_ng_20 = np.load('recovery_rates/m_ng_20.npy') #no_mutator
    m_ng_100 = np.load('recovery_rates/m_ng_100.npy') #no_mutator
    
    f_ng_20 = np.load('recovery_rates/f_ng_20.npy') #full
    f_ng_100 = np.load('recovery_rates/f_ng_100.npy') #full
    
    categories = ['Full', 'PySR', 'No Critic', 'No Planner', 'No Mutator']
    values = np.array([sum(f_ng_20) / 1.2, sum(b_ng_20) / 1.2, sum(c_ng_20) / 1.2, sum(p_ng_20) / 1.2, sum(m_ng_20) / 1.2])
    
    indices = np.argsort(values)
    sorted_names = [categories[i] for i in indices]
    sorted_values = values[indices]
    plt.xlim(0, 100)
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.barh(sorted_names, sorted_values, color='green')
    plt.xlabel('Recovery Rate')
    plt.title('Average recovery rate across 10 runs (sample size 20)')
    plt.savefig('ablation_20.svg')
    plt.savefig('ablation_20')
    plt.show()
    
    values = np.array([sum(f_ng_100) / 1.2, sum(b_ng_100) / 1.2, sum(c_ng_100) / 1.2, sum(p_ng_100) / 1.2, sum(m_ng_100) / 1.2])
    
    indices = np.argsort(values)
    sorted_names = [categories[i] for i in indices]
    sorted_values = values[indices]
    plt.xlim(0, 100)
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.barh(sorted_names, sorted_values, color='orange')
    plt.xlabel('Recovery Rate')
    plt.title('Average recovery rate across 10 runs (sample size 100)')
    plt.savefig('ablation_100.svg')
    plt.savefig('ablation_100')
    plt.show()
    return

_plot_ablation()

_plot_recovery_rates('recovery_rates/b_ng_20.npy', 'recovery_rates/b_ng_100.npy', 
                     'baseline', "Base PySR recovery rates")
_plot_recovery_rates('recovery_rates/c_ng_20.npy', 'recovery_rates/c_ng_100.npy', 
                     'no_critic', "No Critic ablation recovery rates")
_plot_recovery_rates('recovery_rates/p_ng_20.npy', 'recovery_rates/p_ng_100.npy', 
                     'no_planner', "No Planner ablation recovery rates")
_plot_recovery_rates('recovery_rates/m_ng_20.npy', 'recovery_rates/m_ng_100.npy', 
                     'no_mutator', "No Mutator ablation recovery rates")
_plot_recovery_rates('recovery_rates/f_ng_20.npy', 'recovery_rates/f_ng_100.npy', 
                     'full', "Full method recovery rates")