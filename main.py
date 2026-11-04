# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator import Simulator
from policy import LocalizedVQC
from doom import Doom

def run_full_stack(steps=15):
    np.random.seed(42)
    
    vqe_env = Simulator(num_qubits=4)
    vqc_policy = LocalizedVQC(num_qubits=4)
    doom_router = Doom(num_qubits=4)
    
    logs = []
    print("--- Executing Fully Coupled QFusionGuard Simulation ---")
    
    for step in range(1, steps + 1):
        # Step the VQE simulator to get the true plasma Hamiltonian energy
        plasma_hamiltonian_energy = vqe_env.step(lr=0.05)
        
        # put the simulator energy into the VQC input state vector
        env_state = np.full(4, plasma_hamiltonian_energy)
        
        # Update VQC 
        loss = vqc_policy.update(env_state, target_energy=0.0, lr=0.04, noise_std=0.035)
        control_actions = vqc_policy.forward(env_state, noise_std=0.035)
        
        # Derive responsive physical metrics 

        # Q Factor increases as plasma energy decreases toward stable confinement
        q_factor = round(max(1.0, 3.5 - 4.5 * plasma_hamiltonian_energy), 4)
        
        # Core plasma temperature responds dynamically to the energy state and control actions
        temperature_keV = round(16.0 - 2.5 * plasma_hamiltonian_energy + 0.5 * np.mean(control_actions), 4)
        
        # Evaluate threat risk dynamically using DOOM on control actions
        doom_router.update(control_actions, emergency_threshold=0.5, lr=0.04, noise_std=0.035)
        doom_risk_scores = doom_router.forward(control_actions, noise_std=0.035)
        disruption_risk = round(float(np.mean(doom_risk_scores)), 4)
        
        # Determine system status override flag based on dynamic DOOM risk threshold (675)
        system_status = "SAFE_OVERRIDE" if disruption_risk > 0.675 else "NOMINAL_CONTROL"
        
        logs.append({
            "Step": step,
            "plasma_hamiltonian_energy": round(plasma_hamiltonian_energy, 6),
            "Q_Factor": q_factor,
            "Loss": round(loss, 4),
            "Temperature_keV": temperature_keV,
            "Disruption_Risk": disruption_risk,
            "System_Status": system_status
        })
        
    # Convert logs into a DataFrame and write to CSV
    df = pd.DataFrame(logs)
    df.to_csv("qfusion_metrics_telemetry.csv", index=False)
    print("Simulation Complete")
    
    # --- graph ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # Q Plot
    axes[0].plot(df['Step'], df['Q_Factor'], marker='o', color='darkorange', linewidth=2, label='Q-Factor')
    axes[0].set_title('Q Factor', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Q')
    axes[0].legend(loc='upper left')
    
    # Loss Plot
    axes[1].plot(df['Step'], df['Loss'], marker='s', color='teal', linewidth=2, label='VQC Loss')
    axes[1].set_title('VQC Loss', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss Magnitude')
    axes[1].legend(loc='upper right')
    
    # Temp Plot
    axes[2].plot(df['Step'], df['Temperature_keV'], marker='^', color='crimson', linewidth=2, label='Core Plasma Temperature (keV)')
    axes[2].set_title('Temperature Dynamics', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Temperature [keV]')
    axes[2].legend(loc='upper left')
    
    # 4. Disruption Risk Plot
    axes[3].plot(df['Step'], df['Disruption_Risk'], marker='x', color='purple', linewidth=2, label='DOOM Disruption Risk Index')
    axes[3].axhline(y=0.675, color='orange', linestyle='--', label='Override Threshold (0.675)')
    axes[3].set_title('DOOM Disruption Risk', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Simulation Step')
    axes[3].set_ylabel('Risk Index [0, 1]')
    axes[3].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('qfusion_metrics_plots.png', dpi=300, bbox_inches='tight')
    print("Graphs successfully generated")
    
    return df

if __name__ == "__main__":
    df_results = run_full_stack(150)
    print(df_results.to_string())