from pathlib import Path
import pandas as pd
import gc

import matplotlib.pyplot as plt

from BehaviorScreen.core import Stim, Laterality
from .dataset import BehavioralDataLoader
from .point_process import ModelComparator, ModelPlotter
from .poisson_process import RateKernelFactory, PoissonProcess
from .hawkes_process import HistoryKernelFactory, HawkesProcess

def main():
        
    possible_roots = [
        Path('/home/martin/Desktop/DATA'),
        Path('/media/martin/datastore_baier_group/_Projects/Martin_Privat/DATA/Behavioral_screen/DATA/Screen'),
        Path('/media/martin/DATA_18TB/Screen'),
    ]
    ROOT = next((p for p in possible_roots if p.exists()), possible_roots[0])

    # Prey capture stim parameters
    prey_stim_speed_deg_per_s = 90
    prey_stim_range_deg = 2 * 70
    prey_stim_freq = prey_stim_speed_deg_per_s / prey_stim_range_deg

    # 1. Load Data
    loader = BehavioralDataLoader(ROOT / 'bouts_control.csv')

    model_config = {

        'prey_capture_ipsi': {
            'dataset': {
                'stim':Stim.PREY_CAPTURE,
                'bout_name':'JT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity=None)),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A")),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B")),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="rate_shared,gamma")),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="shared")),
                PoissonProcess(RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B,gamma")),
                HawkesProcess(
                    RateKernelFactory.prey_capture(stim_freq=prey_stim_freq, plasticity="A,B,gamma"),
                    HistoryKernelFactory.exponential()
                )
            ]
        },

        'prey_capture_contra': {
            'dataset': {
                'stim':Stim.PREY_CAPTURE,
                'bout_name':'JT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
            ]
        },

        'phototaxis_ipsi': {
            'dataset': {
                'stim':Stim.PHOTOTAXIS,
                'bout_name':'RT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.phototaxis_ipsi()),
            ]
        },

        'phototaxis_contra': {
            'dataset': {
                'stim':Stim.PHOTOTAXIS,
                'bout_name':'RT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':24.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.phototaxis_contra())
            ]
        },

        'omr_lateral_ipsi': {
            'dataset': {
                'epoch_name':["grating right", "grating left"],
                'bout_name':'RT',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson())
            ]
        },

        'omr_lateral_contra': {
            'dataset': {
                'epoch_name':["grating right", "grating left"],
                'bout_name':'RT',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.omr_lateral_contra())
            ]
        },

        'omr_forward': {
            'dataset': {
                'epoch_name':"grating forward",
                'bout_name':'BS',
                'laterality':Laterality.NONDIRECTIONAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.omr_forward())
            ]
        },

        'okr_ipsi': {
            'dataset': {
                'stim':Stim.OKR,
                'bout_name':'S1',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson())
            ]
        },

        'okr_contra': {
            'dataset': {
                'stim':Stim.OKR,
                'bout_name':'S1',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson())
            ]
        },

        'looming_ipsi': {
            'dataset': {
                'stim':Stim.LOOMING,
                'bout_name':'SLC',
                'laterality':Laterality.IPSILATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.looming_gaussian())
            ]
        },

        'looming_contra': {
            'dataset': {
                'stim':Stim.LOOMING,
                'bout_name':'SLC',
                'laterality':Laterality.CONTRALATERAL,
                'binning_dt':0.05, 
                't_start':0.0, 
                't_end':9.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.looming_gaussian())
            ]
        },

        'dark_flash': {
            'dataset': {
                'epoch_name':"flash dark",
                'bout_name':'O',
                'laterality':Laterality.NONDIRECTIONAL,
                'binning_dt':0.025, 
                't_start':0.0, 
                't_end':5.0, 
            },
            'models': [
                PoissonProcess(RateKernelFactory.homogeneous_poisson()),
                PoissonProcess(RateKernelFactory.dark_flash())
            ]
        },
    }

    all_summaries = []

    for exp_name, config in model_config.items():
        
        print(f"\n==================================================")
        print(f" PROCESSING EXPERIMENT: {exp_name.upper()}")
        print(f"==================================================")

        dataset = loader.prepare_dataset(**config['dataset'])

        summary_table, fitted_models = ModelComparator.compare(
            models=config['models'],
            dataset=dataset
        )
        best_model = fitted_models[0]

        summary_table.insert(0, "Condition", exp_name)
        all_summaries.append(summary_table)

        print("\n--- MODEL COMPARISON TABLE ---")
        print(summary_table.to_string(index=False))

        fig1, ax1 = ModelPlotter.plot_model_fits(
            dataset=dataset,
            models=fitted_models,
        )
        plt.show(block=False)

        fig2, axes2 = ModelPlotter.plot_histogram(
            dataset=dataset,
            model=fitted_models[0],
        )
        plt.show(block=False)

        fig3, axes3 = ModelPlotter.plot_trial_traces(
            dataset=dataset,
            model=best_model,
        )
        plt.show(block=False)

        best_model.diagnose(dataset)
        #best_model.bootstrap(dataset, n_boot=500)

        del dataset
        del fitted_models
        del best_model
        del summary_table
        plt.close('all')
        gc.collect()

    master_summary_df = pd.concat(all_summaries, ignore_index=True)
    print("\n================ MASTER MODEL COMPARISON TABLE ================")
    print(master_summary_df.to_string(index=False))

if __name__ == '__main__':
    main()