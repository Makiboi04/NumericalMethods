import numpy as np
import matplotlib.pyplot as plt
from sympy import (symbols, sympify, lambdify, diff, E, exp, Symbol,
                   log, sqrt, Abs, sin, cos, tan, asin, acos, atan, pi)
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re


class NewtonRaphson:
    def __init__(self, master):
        self.master = master
        self.master.title("Newton-Raphson Method")


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
            "derivative": "#bb9af7",
            "tangent": "#e0af68"
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


        style.theme_use("dark_theme")

    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.mainframe, bg=self.colors["bg_dark"])
        header_frame.pack(fill=tk.X, pady=(0, 20))

        title_label = tk.Label(header_frame,
                               text="🔍 Newton-Raphson Method",
                               bg=self.colors["bg_dark"],
                               fg=self.colors["accent"],
                               font=("Segoe UI", 18, "bold"))
        title_label.pack(anchor=tk.W)

        # Add interaction instructions
        instruction_label = tk.Label(header_frame,
                                     text="📱 Interactive Plot: Mouse wheel to zoom • Click and drag to pan • Right-click to reset view",
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
            ("x² - 4", "x**2 - 4"),
            ("sin(x) - x/2", "sin(x) - x/2"),
            ("x³ - 2x - 5", "x**3 - 2*x - 5"),
            ("e^x - 3x", "exp(x) - 3*x"),
            ("ln(x²+1) - 2", "log(x**2 + 1) - 2"),
            ("cos(x) - x·e^(-x)", "cos(x) - x*exp(-x)")
        ]


        for i, (label, func) in enumerate(examples[:3]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func: self.set_example(f)).grid(row=0, column=i, padx=2, sticky="ew")
        for i, (label, func) in enumerate(examples[3:]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func: self.set_example(f)).grid(row=1, column=i, padx=2, pady=(2, 0),
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
        for i in range(3):
            grid_frame.columnconfigure(i * 2 + 1, weight=1)

        # Create parameter inputs
        tk.Label(grid_frame, text="Initial guess x₀:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.x0 = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.x0.grid(row=0, column=1, sticky="ew", padx=(0, 15))
        self.x0.insert(0, "1")

        tk.Label(grid_frame, text="Tolerance:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.tol = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                            fg=self.colors["text_bright"], width=10,
                            insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.tol.grid(row=0, column=3, sticky="ew", padx=(0, 15))
        self.tol.insert(0, "0.0001")

        tk.Label(grid_frame, text="Max Iterations:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=4, sticky="w", padx=(0, 5))
        self.max_iter = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                 fg=self.colors["text_bright"], width=10,
                                 insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.max_iter.grid(row=0, column=5, sticky="ew")
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
        columns = ('n', 'xₙ', 'f(xₙ)', 'f\'(xₙ)', 'Error')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=15)

        # Configure columns
        column_widths = {'n': 40, 'xₙ': 100, 'f(xₙ)': 100, 'f\'(xₙ)': 100, 'Error': 100}

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

        # Enable interactive features
        self.setup_plot_interactions()

    def setup_plot_interactions(self):
        """Setup mouse interactions for the plot"""
        # Initialize interaction variables
        self.pan_active = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.zoom_factor = 1.1

        # Connect mouse events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

        # Store original limits for reset functionality
        self.original_xlim = None
        self.original_ylim = None

    def on_scroll(self, event):
        """Handle mouse wheel scrolling for zooming"""
        if event.inaxes != self.ax:
            return

        # Get current axis limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate zoom center (mouse position)
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        # Determine zoom direction
        if event.button == 'up':
            scale_factor = 1 / self.zoom_factor  # Zoom in
        else:
            scale_factor = self.zoom_factor  # Zoom out

        # Calculate new limits
        x_left = xdata - (xdata - xlim[0]) * scale_factor
        x_right = xdata + (xlim[1] - xdata) * scale_factor
        y_bottom = ydata - (ydata - ylim[0]) * scale_factor
        y_top = ydata + (ylim[1] - ydata) * scale_factor

        # Apply new limits
        self.ax.set_xlim([x_left, x_right])
        self.ax.set_ylim([y_bottom, y_top])

        # Redraw canvas
        self.canvas.draw_idle()

    def on_mouse_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left mouse button - start panning
            self.pan_active = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata

        elif event.button == 3:  # Right mouse button - reset view
            self.reset_plot_view()

    def on_mouse_release(self, event):
        """Handle mouse button release"""
        if event.button == 1:  # Left mouse button
            self.pan_active = False
            self.pan_start_x = None
            self.pan_start_y = None

    def on_mouse_motion(self, event):
        """Handle mouse motion for panning"""
        if not self.pan_active or event.inaxes != self.ax:
            return

        if self.pan_start_x is None or self.pan_start_y is None:
            return

        # Calculate pan distance
        dx = self.pan_start_x - event.xdata
        dy = self.pan_start_y - event.ydata

        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Apply panning
        self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])

        # Redraw canvas
        self.canvas.draw_idle()

    def reset_plot_view(self):
        """Reset plot to original view"""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw_idle()

    def set_example(self, func):
        """Set an example function."""
        self.function.delete(0, tk.END)
        self.function.insert(0, func)

        # Set appropriate initial guesses for each function
        initial_guesses = {
            "x**2 - 4": "3",
            "sin(x) - x/2": "2",
            "x**3 - 2*x - 5": "2",
            "exp(x) - 3*x": "2",
            "log(x**2 + 1) - 2": "1.5",
            "cos(x) - x*exp(-x)": "1"
        }

        if func in initial_guesses:
            self.x0.delete(0, tk.END)
            self.x0.insert(0, initial_guesses[func])

    def parse_function(self, expr_str):
        """Parse the function string to a SymPy expression, robustly handling exponentials and other math functions."""
        try:
            expr_str = expr_str.replace('^', '**')
            # Replace e^x, e^(...), e**x with exp(x)
            expr_str = re.sub(r'e\^$$([^)]+)$$', r'exp(\1)', expr_str)  # e^(...) -> exp(...)
            expr_str = re.sub(r'e\^([a-zA-Z0-9_.]+)', r'exp(\1)', expr_str)  # e^x, e^x.y -> exp(x), exp(x.y)
            expr_str = re.sub(r'e\*\*([a-zA-Z0-9_.]+)', r'exp(\1)', expr_str)  # e**x, e**x.y -> exp(x), exp(x.y)

            # Allow ln as an alias for log (natural logarithm)
            expr_str = expr_str.replace('ln', 'log')

            x = Symbol('x')
            namespace = {
                'x': x,
                'exp': exp,
                'e': E,
                'E': E,
                'log': log,
                'sqrt': sqrt,
                'abs': Abs,
                'sin': sin,
                'cos': cos,
                'tan': tan,
                'asin': asin,
                'acos': acos,
                'atan': atan,
                'pi': pi
            }
            expr = sympify(expr_str, locals=namespace)
            return expr
        except Exception as e:
            raise ValueError(f"Error parsing function: {str(e)}")

    def evaluate_function(self, f, x_val):
        """Safely evaluate a function at a point"""
        try:
            result = float(f(x_val))
            if np.isnan(result) or np.isinf(result):
                raise ValueError("Function evaluation resulted in invalid value")
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating function: {str(e)}")

    def newton_raphson(self, expr, x0, tol, max_iter):
        """Perform Newton-Raphson iteration"""
        x = Symbol('x')
        f = lambdify(x, expr, modules=['numpy'])
        df = lambdify(x, diff(expr, x), modules=['numpy'])
        x_current = x0
        results = []

        for i in range(max_iter):
            try:
                fx = self.evaluate_function(f, x_current)
                dfx = self.evaluate_function(df, x_current)

                if abs(dfx) < 1e-10:
                    raise ValueError("Derivative too close to zero")

                x_next = x_current - fx / dfx
                error = abs((x_next - x_current) / x_next) if x_next != 0 else abs(x_next - x_current)
                results.append((i + 1, x_current, fx, dfx, error))

                if error < tol:
                    break

                x_current = x_next
            except Exception as e:
                messagebox.showerror("Error", f"Error in iteration {i + 1}: {str(e)}")
                return results

        return results

    def calculate(self):
        """Perform calculation and update display"""
        try:
            # Clear previous results
            self.tree.delete(*self.tree.get_children())
            self.ax.clear()

            # Get input values
            expr = self.parse_function(self.function.get())
            x0 = float(self.x0.get())
            tol = float(self.tol.get())
            max_iter = int(self.max_iter.get())

            # Perform Newton-Raphson iteration
            results = self.newton_raphson(expr, x0, tol, max_iter)

            # Update results table with color coding
            for i, result in enumerate(results):
                values = [str(result[0])]  # Iteration number
                values.extend([f"{v:.6f}" if isinstance(v, float) else v for v in result[1:]])

                # Determine tag based on convergence
                error = result[4]
                if error < tol:
                    tag = 'converged'
                elif i > 0 and error < results[i - 1][4]:
                    tag = 'converging'
                else:
                    tag = 'slow_convergence'

                self.tree.insert('', 'end', values=values, tags=(tag,))

            # Enhanced plotting with dark theme
            if results:
                x = Symbol('x')
                f = lambdify(x, expr, modules=['numpy'])
                df = lambdify(x, diff(expr, x), modules=['numpy'])

                # Determine plot range based on iterations
                x_points = [r[1] for r in results]
                x_min = min(x_points) - 2
                x_max = max(x_points) + 2

                x_vals = np.linspace(x_min, x_max, 500)
                try:
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

                    # Plot iteration points and tangent lines
                    for i, result in enumerate(results):
                        x_n, f_xn, df_xn = result[1], result[2], result[3]

                        # Plot iteration point
                        self.ax.plot([x_n], [f_xn], 'o', color=self.colors["interval"],
                                     markersize=8, alpha=0.8, markeredgecolor='white', markeredgewidth=1)

                        # Plot tangent line for visualization
                        if i < len(results) - 1:  # Don't draw tangent for last point
                            tangent_x = np.linspace(x_n - 1, x_n + 1, 100)
                            tangent_y = f_xn + df_xn * (tangent_x - x_n)
                            self.ax.plot(tangent_x, tangent_y, '--', color=self.colors["tangent"],
                                         alpha=0.6, linewidth=1)

                        # Annotate iteration number
                        if i < 5:  # Only annotate first few iterations to avoid clutter
                            self.ax.annotate(f'x{i}', (x_n, f_xn),
                                             xytext=(10, 10), textcoords='offset points',
                                             color=self.colors["text_bright"], fontsize=9,
                                             bbox=dict(boxstyle="round,pad=0.3",
                                                       facecolor=self.colors["bg_light"],
                                                       edgecolor=self.colors["interval"], alpha=0.8))

                    # Highlight final root
                    if results:
                        final = results[-1]
                        self.ax.plot([final[1]], [final[2]], 'o', color=self.colors["success"],
                                     markersize=12, alpha=1.0, markeredgecolor='white', markeredgewidth=2)

                        # Annotate final result
                        self.ax.annotate(f'🎯 Root ≈ {final[1]:.6f}',
                                         (final[1], final[2]),
                                         xytext=(15, 25),
                                         textcoords='offset points',
                                         color=self.colors["text_bright"],
                                         fontsize=12, fontweight='bold',
                                         bbox=dict(boxstyle="round,pad=0.5",
                                                   facecolor=self.colors["bg_light"],
                                                   edgecolor=self.colors["success"],
                                                   alpha=0.9))

                    # Set reasonable y-axis limits
                    y_points = [r[2] for r in results]
                    valid_y = [y for y in y_vals + y_points if not (np.isnan(y) or np.isinf(y))]
                    if valid_y:
                        y_min, y_max = min(valid_y), max(valid_y)
                        y_range = y_max - y_min
                        y_padding = 0.1 * y_range
                        self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

                    # Enhanced plot styling
                    self.ax.set_xlabel('x', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
                    self.ax.set_ylabel('f(x)', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
                    self.ax.set_title('Newton-Raphson Method Visualization',
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

                    self.fig.tight_layout()
                    self.canvas.draw()

                    # Store original limits for reset functionality
                    self.original_xlim = self.ax.get_xlim()
                    self.original_ylim = self.ax.get_ylim()

                    # Show final result
                    if results:
                        final = results[-1]
                        messagebox.showinfo("Success",
                                            f"Root found: {final[1]:.8f}\n"
                                            f"f(root): {final[2]:.8f}\n"
                                            f"Final error: {final[4]:.8f}\n"
                                            f"Iterations: {len(results)}\n"
                                            f"Convergence: {'Achieved' if final[4] < tol else 'Not achieved'}")

                except Exception as e:
                    messagebox.showerror("Error", f"Error plotting function: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear(self):
        """Clear all inputs and results"""
        self.function.delete(0, tk.END)
        self.x0.delete(0, tk.END)
        self.tol.delete(0, tk.END)
        self.max_iter.delete(0, tk.END)
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()
        self.canvas.draw()

        # Reset default values
        self.function.insert(0, "x**2 - 4")
        self.x0.insert(0, "1")
        self.tol.insert(0, "0.0001")
        self.max_iter.insert(0, "50")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Newton-Raphson Method Calculator")

    # Set window size
    window_width = 1400
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app = NewtonRaphson(root)
    root.mainloop()
