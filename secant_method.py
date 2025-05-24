import numpy as np
from sympy import (symbols, sympify, lambdify, E, exp, Symbol, sin, cos, tan,
                   asin, acos, atan, sinh, cosh, tanh, log, sqrt, pi, Abs)
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re


class SecantMethod:
    def __init__(self, master):
        self.master = master
        self.master.title("Secant Method")

        self.colors = {
            "bg_dark": "#1a1b26",
            "bg_light": "#24283b",
            "accent": "#7aa2f7",
            "text": "#a9b1d6",
            "text_bright": "#c0caf5",
            "text_dim": "#565f89",
            "success": "#9ece6a",
            "warning": "#e0af68",
            "error": "#f7768e",
            "function": "#7aa2f7",
            "interval": "#f7768e",
            "midpoint": "#9ece6a",
            "grid": "#414868",
            "secant_line": "#e0af68",
            "iteration_point": "#bb9af7"
        }

        # Configure root window
        self.master.configure(bg=self.colors["bg_dark"])

        # Create main container
        self.mainframe = tk.Frame(self.master, bg=self.colors["bg_dark"])
        self.mainframe.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create header
        self.create_header()

        # Create function input
        self.create_function_input()

        # Create parameters
        self.create_parameters()

        # Create buttons
        self.create_buttons()

        # Configure ttk style for dark theme
        self.configure_ttk_style()

        # Create results section with side-by-side layout
        self.create_results_section()

    def configure_ttk_style(self):
        """Configure ttk style for dark theme"""
        style = ttk.Style()

        # Create custom dark theme for Treeview
        style.theme_create("dark_theme", parent="alt", settings={
            "Treeview": {
                "configure": {
                    "background": self.colors["bg_dark"],
                    "foreground": self.colors["text_bright"],
                    "fieldbackground": self.colors["bg_dark"],
                    "borderwidth": 0
                }
            },
            "Treeview.Heading": {
                "configure": {
                    "background": self.colors["bg_light"],
                    "foreground": self.colors["accent"],
                    "borderwidth": 0,
                    "relief": "flat"
                }
            },
            "Scrollbar": {
                "configure": {
                    "background": self.colors["bg_light"],
                    "troughcolor": self.colors["bg_dark"],
                    "borderwidth": 0
                }
            }
        })

        # Use the custom theme
        style.theme_use("dark_theme")

    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(header_frame,
                               text="🔍 Secant Method",
                               bg=self.colors["bg_dark"],
                               fg=self.colors["accent"],
                               font=("Segoe UI", 18, "bold"))
        title_label.pack(anchor=tk.W)

        # Add interaction instructions
        instruction_label = tk.Label(header_frame,
                                     text="📱 Interactive Plot: Derivative-free convergence using secant line approximations",
                                     bg=self.colors["bg_dark"],
                                     fg=self.colors["text_dim"],
                                     font=("Segoe UI", 9))
        instruction_label.pack(anchor=tk.W, pady=(5, 0))

    def create_function_input(self):
        """Create function input section"""
        input_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        input_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(input_frame, text="Function f(x):",
                 bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        self.function = tk.Entry(input_frame,
                                 bg=self.colors["bg_light"],
                                 fg=self.colors["text_bright"],
                                 insertbackground=self.colors["accent"],
                                 font=("Segoe UI", 10),
                                 relief="flat", bd=5)
        self.function.pack(fill=tk.X, pady=(0, 10))
        self.function.insert(0, "x**2 - 4")

        # Example functions
        tk.Label(input_frame, text="Examples:",
                 bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(5, 5))

        examples_frame = tk.Frame(input_frame, bg=self.colors["bg_dark"])
        examples_frame.pack(fill=tk.X)

        examples = [
            ("x² - 4", "x**2 - 4", "-3", "-1"),
            ("sin(x) - x/2", "sin(x) - x/2", "1", "2"),
            ("x³ - 2x - 5", "x**3 - 2*x - 5", "2", "3"),
            ("e^x - 3x", "exp(x) - 3*x", "0", "1"),
            ("ln(x²+1) - 2", "log(x**2 + 1) - 2", "2.5", "3.0"),
            ("cos(x) - x·e^(-x)", "cos(x) - x*exp(-x)", "0", "1")
        ]

        # Create two rows of example buttons
        for i, (label, func, start_x0, start_x1) in enumerate(examples[:3]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, x0=start_x0, x1=start_x1: self.set_example(f, x0, x1)).grid(row=0,
                                                                                                         column=i,
                                                                                                         padx=2,
                                                                                                         sticky="ew")
        for i, (label, func, start_x0, start_x1) in enumerate(examples[3:]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, x0=start_x0, x1=start_x1: self.set_example(f, x0, x1)).grid(row=1,
                                                                                                         column=i,
                                                                                                         padx=2,
                                                                                                         pady=(2, 0),
                                                                                                         sticky="ew")

        # Configure grid weights for examples
        for i in range(3):
            examples_frame.columnconfigure(i, weight=1)

    def create_parameters(self):
        """Create parameters section"""
        params_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        params_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(params_frame, text="Parameters:",
                 bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

        # Parameters grid
        grid_frame = tk.Frame(params_frame, bg=self.colors["bg_dark"])
        grid_frame.pack(fill=tk.X)

        # Configure grid
        for i in range(4):
            grid_frame.columnconfigure(i, weight=1)

        # Create parameter inputs
        tk.Label(grid_frame, text="Initial x₀:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.x0 = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.x0.grid(row=0, column=1, sticky="ew", padx=(0, 15))
        self.x0.insert(0, "-3")

        tk.Label(grid_frame, text="Initial x₁:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.x1 = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.x1.grid(row=0, column=3, sticky="ew")
        self.x1.insert(0, "-1")

        tk.Label(grid_frame, text="Tolerance:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        self.tol = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                            fg=self.colors["text_bright"], width=10,
                            insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.tol.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=(10, 0))
        self.tol.insert(0, "0.0001")

        tk.Label(grid_frame, text="Max Iterations:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(10, 0))
        self.max_iter = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                 fg=self.colors["text_bright"], width=10,
                                 insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.max_iter.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.max_iter.insert(0, "50")

    def create_buttons(self):
        """Create buttons section"""
        buttons_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        buttons_frame.pack(fill=tk.X, pady=(0, 20))

        self.calculate_btn = tk.Button(buttons_frame, text="🚀 Calculate",
                                       bg=self.colors["accent"], fg=self.colors["bg_dark"],
                                       font=("Segoe UI", 10, "bold"),
                                       relief="flat", bd=0, padx=20, pady=8,
                                       command=self.calculate)
        self.calculate_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_btn = tk.Button(buttons_frame, text="🗑️ Clear",
                                   bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                                   font=("Segoe UI", 10),
                                   relief="flat", bd=0, padx=20, pady=8,
                                   command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

    def create_results_section(self):
        """Create results section with side-by-side layout"""
        results_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid for side-by-side layout
        results_frame.columnconfigure(0, weight=1)  # Table column
        results_frame.columnconfigure(1, weight=2)  # Plot column (expanded)
        results_frame.rowconfigure(0, weight=1)

        # Table section (left side)
        table_frame = tk.Frame(results_frame, bg=self.colors["bg_dark"])
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(table_frame, text="📊 Iteration Results:",
                 bg=self.colors["bg_dark"], fg=self.colors["accent"],
                 font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Create table with scrollbar
        table_container = tk.Frame(table_frame, bg=self.colors["bg_dark"])
        table_container.pack(fill=tk.BOTH, expand=True)

        # Define columns
        columns = ('n', 'xₙ₋₁', 'xₙ', 'f(xₙ₋₁)', 'f(xₙ)', 'xₙ₊₁', 'Error')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=15)

        # Configure columns
        column_widths = {'n': 40, 'xₙ₋₁': 80, 'xₙ': 80, 'f(xₙ₋₁)': 80, 'f(xₙ)': 80, 'xₙ₊₁': 80, 'Error': 80}

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')

        # Configure row colors for convergence visualization
        self.tree.tag_configure('converged',
                                background=self.colors["success"],
                                foreground='black')
        self.tree.tag_configure('converging',
                                background=self.colors["bg_light"],
                                foreground=self.colors["text_bright"])
        self.tree.tag_configure('slow_convergence',
                                background=self.colors["warning"],
                                foreground='black')

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Plot section (right side - expanded)
        plot_frame = tk.Frame(results_frame, bg=self.colors["bg_dark"])
        plot_frame.grid(row=0, column=1, sticky="nsew")

        plot_header = tk.Frame(plot_frame, bg=self.colors["bg_dark"])
        plot_header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(plot_header, text="📈 Interactive Function Visualization:",
                 bg=self.colors["bg_dark"], fg=self.colors["accent"],
                 font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor=self.colors["bg_dark"])
        self.ax = self.fig.add_subplot(111, facecolor=self.colors["bg_dark"])

        # Style the plot
        self.ax.tick_params(colors=self.colors["text"], which='both')
        for spine in self.ax.spines.values():
            spine.set_color(self.colors["grid"])

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def set_example(self, func, x0, x1):
        """Set an example function with appropriate initial points."""
        self.function.delete(0, tk.END)
        self.function.insert(0, func)
        self.x0.delete(0, tk.END)
        self.x0.insert(0, x0)
        self.x1.delete(0, tk.END)
        self.x1.insert(0, x1)

    def parse_function(self, expr_str):
        """Parse the function string to a SymPy expression."""
        try:
            # Replace ^ with ** for exponentiation
            expr_str = expr_str.replace('^', '**')

            # Handle exponentials
            expr_str = re.sub(r'e\^$$([^)]+)$$', r'exp(\1)', expr_str)
            expr_str = re.sub(r'e\^([a-zA-Z0-9_.]+)', r'exp(\1)', expr_str)
            expr_str = re.sub(r'e\*\*([a-zA-Z0-9_.]+)', r'exp(\1)', expr_str)

            # Allow ln as alias for log
            expr_str = expr_str.replace('ln', 'log')

            x = symbols('x')
            namespace = {
                'x': x, 'exp': exp, 'e': E, 'E': E,
                'log': log, 'sqrt': sqrt,
                'abs': Abs, 'sin': sin, 'cos': cos, 'tan': tan,
                'asin': asin, 'acos': acos, 'atan': atan, 'pi': pi
            }
            return sympify(expr_str, locals=namespace)
        except Exception as e:
            raise ValueError(f"Error parsing function: {str(e)}")

    def evaluate_function(self, f, x_val):
        """Safely evaluate a function at a point."""
        try:
            result = float(f(x_val))
            if np.isnan(result) or np.isinf(result):
                raise ValueError("Function evaluation resulted in invalid value")
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating function at x={x_val}: {str(e)}")

    def secant_method(self, expr, x0, x1, tol, max_iter):
        """Perform Secant Method iteration."""
        x = symbols('x')
        f = lambdify(x, expr, modules=['numpy'])
        results = []

        fx0 = self.evaluate_function(f, x0)
        fx1 = self.evaluate_function(f, x1)

        for i in range(max_iter):
            if abs(fx1 - fx0) < 1e-10:
                raise ValueError("Division by zero encountered")

            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            error = abs((x2 - x1) / x2) if x2 != 0 else abs(x2 - x1)

            results.append((i + 1, x0, x1, fx0, fx1, x2, error))

            if abs(fx1) < tol or error < tol:
                break

            x0, x1 = x1, x2
            fx0, fx1 = fx1, self.evaluate_function(f, x2)

        return results

    def calculate(self):
        """Perform calculation and update display."""
        try:
            # Clear previous results
            self.tree.delete(*self.tree.get_children())
            self.ax.clear()

            # Get input values
            expr = self.parse_function(self.function.get())
            x0 = float(self.x0.get())
            x1 = float(self.x1.get())
            tol = float(self.tol.get())
            max_iter = int(self.max_iter.get())

            # Perform Secant iteration
            results = self.secant_method(expr, x0, x1, tol, max_iter)

            # Update results table with color coding
            for i, result in enumerate(results):
                values = [str(result[0])]  # Iteration number
                values.extend([f"{v:.6f}" if isinstance(v, float) else v for v in result[1:]])

                # Determine tag based on convergence
                error = result[6]
                if error < tol:
                    tag = 'converged'
                elif i > 0 and error < results[i - 1][6]:
                    tag = 'converging'
                else:
                    tag = 'slow_convergence'

                self.tree.insert('', 'end', values=values, tags=(tag,))

            # Enhanced plotting with dark theme
            x = symbols('x')
            f = lambdify(x, expr, modules=['numpy'])

            # Determine plot range
            x_points = [r[1] for r in results] + [r[2] for r in results] + [r[5] for r in results]
            x_min, x_max = min(x_points), max(x_points)
            x_range = x_max - x_min
            x_min -= 0.5 * x_range
            x_max += 0.5 * x_range

            # Create smooth curve with more points for better resolution
            x_vals = np.linspace(x_min, x_max, 500)
            y_vals = [self.evaluate_function(f, x) for x in x_vals]

            # Style plot with dark theme
            self.ax.set_facecolor(self.colors["bg_dark"])
            self.ax.tick_params(colors=self.colors["text"], which='both', labelsize=11)

            # Plot function with enhanced styling
            self.ax.plot(x_vals, y_vals, '-', color=self.colors["function"],
                         linewidth=3, label='f(x)', alpha=0.9)
            self.ax.axhline(y=0, color=self.colors["grid"], linestyle='--', alpha=0.7, linewidth=2)
            self.ax.axvline(x=0, color=self.colors["grid"], linestyle='--', alpha=0.3)
            self.ax.grid(True, alpha=0.3, color=self.colors["grid"], linewidth=1)

            # Plot iteration points and secant lines with enhanced visibility
            for i, result in enumerate(results):
                x0_val, x1_val, _, fx0, fx1, x2 = result[1:7]

                # Plot points with different sizes for emphasis
                self.ax.plot([x0_val], [fx0], 'o', color=self.colors["interval"],
                             markersize=8, alpha=0.8, markeredgecolor='white', markeredgewidth=1)
                self.ax.plot([x1_val], [fx1], 'o', color=self.colors["interval"],
                             markersize=8, alpha=0.8, markeredgecolor='white', markeredgewidth=1)

                # Plot secant line with enhanced styling
                line_x = np.array([x0_val, x1_val])
                line_y = np.array([fx0, fx1])
                self.ax.plot(line_x, line_y, '--', color=self.colors["secant_line"],
                             alpha=0.7, linewidth=2, label='Secant lines' if i == 0 else "")

                # Plot next point (intersection with x-axis)
                fx2 = self.evaluate_function(f, x2)
                if i == len(results) - 1:  # Last iteration - highlight final result
                    self.ax.plot([x2], [fx2], 'o', color=self.colors["success"],
                                 markersize=12, alpha=1.0, markeredgecolor='white', markeredgewidth=2)

                    # Annotate final result
                    self.ax.annotate(f'🎯 Root ≈ {x2:.6f}',
                                     (x2, fx2),
                                     xytext=(15, 25),
                                     textcoords='offset points',
                                     color=self.colors["text_bright"],
                                     fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor=self.colors["bg_light"],
                                               edgecolor=self.colors["success"],
                                               alpha=0.9))
                else:
                    self.ax.plot([x2], [fx2], 'o', color=self.colors["iteration_point"],
                                 markersize=6, alpha=0.8)

                # Annotate first few iterations
                if i < 3:  # Only annotate first few iterations to avoid clutter
                    self.ax.annotate(f'Iter {result[0]}', (x1_val, fx1),
                                     xytext=(10, 10), textcoords='offset points',
                                     color=self.colors["text_bright"], fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor=self.colors["bg_light"],
                                               edgecolor=self.colors["secant_line"], alpha=0.8))

            # Set reasonable y-axis limits with more margin
            y_points = [y for y in y_vals if not (np.isinf(y) or np.isnan(y))]
            if y_points:
                y_min, y_max = min(y_points), max(y_points)
                y_range = y_max - y_min
                y_padding = 0.2 * y_range
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Enhanced plot styling
            self.ax.set_xlabel('x', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_ylabel('f(x)', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_title('Secant Method Visualization',
                              color=self.colors["text_bright"], fontsize=14, fontweight='bold', pad=20)

            # Style spines
            for spine in self.ax.spines.values():
                spine.set_color(self.colors["grid"])
                spine.set_linewidth(1.5)

            # Legend with dark theme
            legend = self.ax.legend(loc='best', fontsize=10)
            legend.get_frame().set_facecolor(self.colors["bg_light"])
            legend.get_frame().set_edgecolor(self.colors["grid"])
            for text in legend.get_texts():
                text.set_color(self.colors["text_bright"])

            # Adjust layout to prevent cutting off labels
            self.fig.tight_layout()
            self.canvas.draw()

            # Show final result
            if results:
                final = results[-1]
                messagebox.showinfo("Success",
                                    f"Root found: {final[5]:.8f}\n"
                                    f"f(root): {self.evaluate_function(f, final[5]):.8f}\n"
                                    f"Final error: {final[6]:.8f}\n"
                                    f"Iterations: {len(results)}\n"
                                    f"Convergence: {'Achieved' if final[6] < tol else 'Not achieved'}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear(self):
        """Clear all inputs and results."""
        self.function.delete(0, tk.END)
        self.x0.delete(0, tk.END)
        self.x1.delete(0, tk.END)
        self.tol.delete(0, tk.END)
        self.max_iter.delete(0, tk.END)
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()
        self.canvas.draw()

        # Reset default values
        self.function.insert(0, "x**2 - 4")
        self.x0.insert(0, "-3")
        self.x1.insert(0, "-1")
        self.tol.insert(0, "0.0001")
        self.max_iter.insert(0, "50")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Secant Method Calculator")

    # Set window size
    window_width = 1400
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app = SecantMethod(root)
    root.mainloop()
