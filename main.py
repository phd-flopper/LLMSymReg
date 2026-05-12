import juliacall
from runNguyen import run_Nguyen
from runBaseline import run_all
from runFeynman import run_Feynman

def main():
    run_Nguyen()
    run_all()
    run_Feynman()

if __name__ == "__main__":
    main()