import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils.qft import QFT
from utils.iqft import IQFT
from utils.phase_estimation import PhaseEstimation
from qiskit.visualization import plot_bloch_multivector, plot_distribution, plot_state_city, plot_histogram
from qiskit.quantum_info import state_fidelity

class QFTSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quantum Algorithm Simulator")
        self.geometry("1300x800")
        self.create_menu()
        self.create_layout()
        self.circuit_img = None
        self.noise_type = None
        self.noise_params = None
        self.current_simulation = None

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Circuit Image", command=self.save_circuit)
        file_menu.add_command(label="Save Results", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Input", command=self.clear_input)
        edit_menu.add_command(label="Reset Simulation", command=self.reset_simulation)
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Bloch Sphere Analysis", command=self.show_bloch_sphere_window, state=tk.DISABLED)
        analysis_menu.add_command(label="Probability Analysis", command=self.show_probability_window, state=tk.DISABLED)
        analysis_menu.add_command(label="State City Analysis", command=self.show_state_city_window, state=tk.DISABLED)
        analysis_menu.add_command(label="Comparison Analysis", command=self.show_comparison_window, state=tk.DISABLED)
        self.analysis_menu = analysis_menu

    def create_layout(self):
        # Control panel
        self.create_control_panel()
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.create_left_panel(main_frame)
        self.create_center_panel(main_frame)
        self.create_bottom_panel(main_frame)
        self.add_text_to_bottom_panel("Simulator ready")
        self.current_result = None

    def create_control_panel(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(control_frame, text="Algorithm:", font=("Arial", 11, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.algorithm_var = tk.StringVar(value="QFT")
        algorithms = ["QFT", "IQFT", "Phase Estimation"]
        self.algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var, values=algorithms, state="readonly")
        self.algorithm_combo.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(control_frame, text="Noise:", font=("Arial", 11, "bold")).grid(row=0, column=2, padx=15, pady=5, sticky='w')
        self.noise_var = tk.BooleanVar()
        self.noise_check = ttk.Checkbutton(control_frame, text="Enable", variable=self.noise_var)
        self.noise_check.grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(control_frame, text="Type of noise:", font=("Arial", 11, "bold")).grid(row=0, column=4, padx=5, pady=5, sticky='w')
        self.noise_type_var = tk.StringVar(value="Bit flip")
        noise_types = ["Bit flip", "Phase flip", "Depolarizing", "Phase Damping", "Amplitude Damping"]
        self.noise_type_combo = ttk.Combobox(control_frame, textvariable=self.noise_type_var, values=noise_types, state="readonly")
        self.noise_type_combo.grid(row=0, column=5, padx=5, pady=5)
        self.add_noise = ttk.Button(control_frame, text="Noise Settings", command=self.create_noise_panel)
        self.add_noise.grid(row=0, column=6, padx=5, pady=5)
        ttk.Label(control_frame, text="Options:", font=("Arial", 11, "bold")).grid(row=0, column=7, padx=5, pady=5, sticky='w')
        self.measure = tk.BooleanVar()
        self.measurements = ttk.Checkbutton(control_frame, text="Measure", variable=self.measure)
        self.measurements.grid(row=0, column=8, padx=5, pady=5)
        self.compare_var = tk.BooleanVar()
        self.compare = ttk.Checkbutton(control_frame, text="Comparison", variable=self.compare_var)
        self.compare.grid(row=0, column=9, padx=5, pady=5)

    def create_left_panel(self, parent):
        left_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        ttk.Label(left_frame, text="Number of input qubits (n):", font=("Arial", 10)).pack(pady=5)
        self.n_entry = ttk.Entry(left_frame, width=15, justify='center')
        self.n_entry.pack(pady=5)
        ttk.Button(left_frame, text="Set Qubit States", command=self.generate_qubit_states).pack(pady=20)
        self.state_selectors_frame = ttk.Frame(left_frame)
        self.state_selectors_frame.pack()
        ttk.Label(left_frame, text="Build and run", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Button(left_frame, text="Build Quantum Circuit", command=self.build_circuit).pack(pady=10)
        ttk.Button(left_frame, text="Run Simulation", command=self.run_simulation).pack(pady=10)

    def generate_qubit_states(self):
        for widget in self.state_selectors_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()
        if algorithm == "QFT":
            try:
                n = int(self.n_entry.get())
                if n <= 0:
                    raise ValueError
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid positive integer for number of qubits.")
                return
            self.qubit_states = []
            basis_states = ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "|i⟩", "|−i⟩"]
            for i in range(n):
                label = tk.Label(self.state_selectors_frame, text=f"Qubit {i} state:")
                label.grid(row=i, column=0, padx=5, pady=2)

                state_var = tk.StringVar()
                state_var.set(basis_states[0])
                dropdown = ttk.Combobox(self.state_selectors_frame, textvariable=state_var, values=basis_states, state="readonly", width=6)
                dropdown.grid(row=i, column=1, padx=5, pady=2)

                self.qubit_states.append(state_var)
        elif algorithm == "IQFT":
            label = tk.Label(self.state_selectors_frame, text="Enter state number:")
            label.grid(row=0, column=0, padx=5, pady=2)
            self.iqft_number_entry = tk.Entry(self.state_selectors_frame, width=15, justify='center')
            self.iqft_number_entry.grid(row=0, column=1, padx=5, pady=2)
        elif algorithm == "Phase Estimation":
            return
        else:
            try:
                n = int(self.n_entry.get())
                if n <= 0:
                    raise ValueError
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Please enter a valid positive integer for number of qubits.")
                return
            self.qubit_states = []
            basis_states = ["|0⟩", "|1⟩", "|+⟩", "|−⟩", "|i⟩", "|−i⟩"]
            for i in range(n):
                label = tk.Label(self.state_selectors_frame, text=f"Qubit {i} state:")
                label.grid(row=i, column=0, padx=5, pady=2)

                state_var = tk.StringVar()
                state_var.set(basis_states[0])
                dropdown = ttk.Combobox(self.state_selectors_frame, textvariable=state_var, values=basis_states, state="readonly", width=6)
                dropdown.grid(row=i, column=1, padx=5, pady=2)

                self.qubit_states.append(state_var)

    def create_noise_panel(self):
        window = tk.Toplevel(self)
        window.title("Noise Parameters")
        window.geometry("300x120")
        
        noise_frame = ttk.LabelFrame(window, text="Noise Parameters", padding=10)
        noise_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        noise_type = self.noise_type_var.get()
        param_label = ttk.Label(noise_frame, text="Parameter:")
        param_label.grid(row=0, column=0, padx=5, pady=5)
        param_var = tk.DoubleVar(value=0.01)
        param_entry = ttk.Entry(noise_frame, textvariable=param_var, width=10)
        param_entry.grid(row=0, column=1, padx=5, pady=5)
        if noise_type == "Bit flip":
            param_label.config(text="Probability of bit flip:")
        elif noise_type == "Phase flip":
            param_label.config(text="Probability of phase flip:")
        elif noise_type == "Depolarizing":
            param_label.config(text="Lambda for depolarization 0<=λ<4^n/(4^n-1):")
        elif noise_type == "Phase Damping":
            param_label.config(text="Phase damping parameter:")
        elif noise_type == "Amplitude Damping":
            param_label.config(text="Amplitude damping parameter:")
        def apply():
            self.noise_params = {noise_type.lower().replace(' ', '_'): param_var.get()}
            self.status_label.config(text="Noise applied!")
            window.destroy()
        ttk.Button(noise_frame, text="Apply Noise", command=apply).grid(row=2, column=0, columnspan=2, pady=10)
        self.noise_type = noise_type

    def create_center_panel(self, parent):
        center_frame = ttk.LabelFrame(parent, text="Circuit Visualization", padding=10)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.circuit_canvas = tk.Canvas(center_frame, width=600, height=400, highlightthickness=0)
        self.circuit_canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.status_label = ttk.Label(center_frame, text="Ready to build circuit", font=("Arial", 11), background="#eaf0fa")
        self.status_label.pack(pady=5)

    def create_bottom_panel(self, parent):
        bottom_frame = ttk.LabelFrame(parent, text="Simulation Output", padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=10, pady=10)
        self.output_text = tk.Text(bottom_frame, height=8, wrap=tk.WORD, font=("Courier", 11), relief=tk.FLAT)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.config(state=tk.DISABLED)

    def add_text_to_bottom_panel(self, text):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, text)
        self.output_text.insert(tk.END, "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)

    def clear_bottom_panel(self):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def show_bloch_sphere_window(self):
        if self.current_simulation is None or self.current_simulation.get_resulting_state() is None:
            messagebox.showwarning("Warning", "No simulation results available. Run a simulation first.")
            return
        
        window = tk.Toplevel(self)
        window.title("Bloch Sphere Analysis")
        window.geometry("800x600")

        state = self.current_simulation.get_resulting_state()
        
        self.add_text_to_bottom_panel(state)

        fig = plot_bloch_multivector(state, figsize=(8, 6), title='Bloch Spheres', reverse_bits=True)

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()

    def show_probability_window(self):
        if self.current_simulation.get_resulting_state() is None:
            messagebox.showwarning("Warning", "No simulation results available. Run a simulation first.")
            return
        
        window = tk.Toplevel(self)
        window.title("Probability Analysis")
        window.geometry("800x600")

        counts = self.current_simulation.get_resulting_counts()

        fig = plot_distribution(counts, figsize=(8, 6), title='Probability Distribution')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()

    def show_state_city_window(self):
        if self.current_simulation.get_resulting_state() is None:
            messagebox.showwarning("Warning", "No simulation results available. Run a simulation first.")
            return
            
        window = tk.Toplevel(self)
        window.title("State City Analysis")
        window.geometry("800x600")
        
        counts = self.current_simulation.get_resulting_state()

        fig = plot_state_city(counts, figsize=(8, 6), title='State City')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas.draw()
          
    def show_comparison_window(self):
        if self.current_simulation.get_circuit() is None:
            messagebox.showwarning("Warning", "No simulation results available. Run a simulation first.")
            return
        
        window = tk.Toplevel(self)
        window.title("Fidelity Analysis")
        window.geometry("800x600")

        ideal_counts, noisy_counts = self.current_simulation.get_comp_counts()
        ideal_fidelity, noisy_fidelity = self.current_simulation.get_comp_fidelity()

        fidelity = state_fidelity(ideal_fidelity, noisy_fidelity)
        self.add_text_to_bottom_panel(f"Fidelity: {fidelity:.4f}")

        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        plot_histogram([ideal_counts, noisy_counts],
                    legend=['Ideal', 'Noisy'],
                    ax=ax,
                    title='Count Comparison')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def build_circuit(self):
        self.add_text_to_bottom_panel("Building quantum circuit...")
        n_str = self.n_entry.get()
        
        try:
            n = int(n_str)
            if n < 1 or n > 10:
                raise ValueError("Number of qubits must be between 1 and 10")

            algorithm = self.algorithm_var.get()
            if algorithm == "QFT":
                state = ""
                
                for i in range(len(self.qubit_states)):
                    temp = str(self.qubit_states[i].get())
                    if temp == "|0⟩":
                        state += "0"
                    elif temp == "|1⟩":
                        state += "1"
                    elif temp == "|+⟩":
                        state += "+"
                    elif temp == "|−⟩":
                        state += "-"
                    elif temp == "|i⟩":
                        state += "r"
                    elif temp == "|−i⟩":
                        state += "l"
                    else:
                        messagebox.showerror("Error", f"Invalid state for qubit {i+1}: {temp}")
                        return
                self.current_simulation = QFT(n=n, state=state, noise=self.noise_var.get(), noise_type=self.noise_type, noise_options=self.noise_params, measure=self.measure.get(), comparison=self.compare_var.get())
                img_path = self.current_simulation.build_qft()
            elif algorithm == "IQFT":
                state = self.iqft_number_entry.get()
                self.current_simulation = IQFT(n=n, state=state, noise=self.noise_var.get(), noise_type=self.noise_type, noise_options=self.noise_params, measure=self.measure.get(), comparison=self.compare_var.get())
                img_path = self.current_simulation.build_iqft()
            elif algorithm == "Phase Estimation":
                self.current_simulation = PhaseEstimation(n=n, state=None, noise=self.noise_var.get(), noise_type=self.noise_type, noise_options=self.noise_params, measure=self.measure.get(), comparison=self.compare_var.get())
                img_path = self.current_simulation.build_phase_estimation()
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self.display_circuit(img_path)
            
            self.status_label.config(text=f"Circuit built with {n} qubits")
            
            self.add_text_to_bottom_panel(f"Circuit built with {n} qubits using {algorithm} algorithm")
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def display_circuit(self, img_path):
        try:
            img = Image.open(img_path)
            orig_width, orig_height = img.size
            
            canvas_width = self.circuit_canvas.winfo_width()
            canvas_height = self.circuit_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                max_width = canvas_width - 40
                max_height = canvas_height - 40
                
                scale_width = max_width / orig_width
                scale_height = max_height / orig_height
                scale_factor = min(scale_width, scale_height)
                
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                img.thumbnail((600, 300), Image.Resampling.LANCZOS)
            
            self.circuit_img = ImageTk.PhotoImage(img)
            self.circuit_canvas.delete("all")
            
            canvas_center_x = canvas_width // 2 if canvas_width > 1 else 200
            canvas_center_y = canvas_height // 2 if canvas_height > 1 else 150
            
            self.circuit_canvas.create_image(canvas_center_x, canvas_center_y, image=self.circuit_img)
            self.last_img_path = img_path
        except Exception as e:
            self.status_label.config(text=f"Error displaying circuit: {e}")

    def run_simulation(self):
        self.add_text_to_bottom_panel("Running simulation...")
        try:
            self.current_simulation.simulate()

            if self.compare_var.get():
                self.analysis_menu.entryconfig("Comparison Analysis", state=tk.NORMAL)
                self.analysis_menu.entryconfig("Bloch Sphere Analysis", state=tk.DISABLED)
                self.analysis_menu.entryconfig("Probability Analysis", state=tk.DISABLED)
                self.analysis_menu.entryconfig("State City Analysis", state=tk.DISABLED)
            else:
                self.analysis_menu.entryconfig("Bloch Sphere Analysis", state=tk.NORMAL)
                if self.measure.get():
                    self.analysis_menu.entryconfig("Probability Analysis", state=tk.NORMAL)  
                    self.analysis_menu.entryconfig("State City Analysis", state=tk.NORMAL)
                else:
                    self.analysis_menu.entryconfig("Probability Analysis", state=tk.DISABLED)
                    self.analysis_menu.entryconfig("State City Analysis", state=tk.DISABLED)
                self.analysis_menu.entryconfig("Comparison Analysis", state=tk.DISABLED)
            self.status_label.config(text="Simulation completed - Analysis menu enabled")
            
            self.add_text_to_bottom_panel("Simulation completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")

    def save_circuit(self):
        if hasattr(self, 'last_img_path') and self.last_img_path:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if filename:
                import shutil
                shutil.copy(self.last_img_path, filename)
                messagebox.showinfo("Success", f"Circuit saved to {filename}")
        else:
            messagebox.showwarning("Warning", "No circuit to save. Build a circuit first.")

    def save_results(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(f"Algorithm: {self.algorithm_var.get()}\n")
                f.write(f"Qubits: {self.n_entry.get()}\n")
                f.write(f"Noise: {self.noise_var.get()}\n")
            messagebox.showinfo("Success", f"Results saved to {filename}")

    def clear_input(self):
        self.n_entry.delete(0, tk.END)
        self.qubit_states = []
        self.noise_var.set(False)
        self.noise_type_var.set("Bit flip")
        self.noise_params = None
        self.state_selectors_frame.destroy()

    def reset_simulation(self):
        self.analysis_menu.entryconfig("Bloch Sphere Analysis", state=tk.DISABLED)
        self.analysis_menu.entryconfig("Probability Analysis", state=tk.DISABLED)
        self.analysis_menu.entryconfig("State City Analysis", state=tk.DISABLED)
        self.analysis_menu.entryconfig("Comparison Analysis", state=tk.DISABLED)
        self.clear_bottom_panel()
        self.status_label.config(text="Ready to build circuit")

if __name__ == "__main__":
    app = QFTSimulator()
    app.mainloop()