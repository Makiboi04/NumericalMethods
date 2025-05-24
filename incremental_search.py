import numpy as np
from sympy import (symbols, sympify, lambdify, E, exp, Symbol,
                   log, sqrt, Abs, sin, cos, tan, asin, acos, atan, pi)
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re


class IncrementalSearch:
    def __init__(self, master):
        self.master = master
        self.master.title("Incremental Search Method")

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
            "first_subinterval": "#152A4D",
            "second_subinterval": "#295396"
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
                               text="🔍 Incremental Search Method",
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
            ("x² - 4", "x**2 - 4", "-3", "3", "0.5"),
            ("sin(x) - x/2", "sin(x) - x/2", "0", "4", "0.2"),
            ("x³ - 2x - 5", "x**3 - 2*x - 5", "0", "5", "0.2"),
            ("e^x - 3x", "exp(x) - 3*x", "0", "2", "0.1"),
            ("ln(x²+1) - 2", "log(x**2 + 1) - 2", "2", "3", "0.1"),
            ("cos(x) - x·e^(-x)", "cos(x) - x*exp(-x)", "0", "2", "0.1")
        ]

        # Create two rows of example buttons
        for i, (label, func, start_x, _, delta_x) in enumerate(examples[:3]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, start=start_x, delta=delta_x: self.set_example(f, start, delta)).grid(
                row=0, column=i, padx=2, sticky="ew")
        for i, (label, func, start_x, _, delta_x) in enumerate(examples[3:]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, start=start_x, delta=delta_x: self.set_example(f, start, delta)).grid(
                row=1, column=i, padx=2, pady=(2, 0), sticky="ew")

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
        tk.Label(grid_frame, text="Initial x₁:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.x1 = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.x1.grid(row=0, column=1, sticky="ew", padx=(0, 15))
        self.x1.insert(0, "-3")

        tk.Label(grid_frame, text="Delta x:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.dx = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.dx.grid(row=0, column=3, sticky="ew")
        self.dx.insert(0, "0.5")

        tk.Label(grid_frame, text="Max Iterations:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        self.max_iter = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                 fg=self.colors["text_bright"], width=10,
                                 insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.max_iter.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=(10, 0))
        self.max_iter.insert(0, "50")

        tk.Label(grid_frame, text="Tolerance:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(10, 0))
        self.tol = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                            fg=self.colors["text_bright"], width=10,
                            insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.tol.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.tol.insert(0, "0.0001")

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
        columns = ('Iteration', 'x₁', 'Δx', 'x₂', 'f(x₁)', 'f(x₂)', 'f(x₁)·f(x₂)', 'Remark')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=15)

        # Configure columns
        column_widths = {'Iteration': 60, 'x₁': 70, 'Δx': 50, 'x₂': 70,
                         'f(x₁)': 70, 'f(x₂)': 70, 'f(x₁)·f(x₂)': 80, 'Remark': 120}

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')

        # Configure row colors for different remarks
        self.tree.tag_configure('root_found',
                                background=self.colors["success"],
                                foreground='black')
        self.tree.tag_configure('sign_change',
                                background=self.colors["warning"],
                                foreground='black')
        self.tree.tag_configure('continue',
                                background=self.colors["bg_light"],
                                foreground=self.colors["text_bright"])

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

            x = Symbol('x')
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

    def incremental_search(self, expr, x1, dx, max_iter, tol):
        """
        Perform Incremental Search Method iteration."""
        x = Symbol('x')
        f = lambdify(x, expr, modules=['numpy'])
        results = []

        x_current = x1

        for i in range(max_iter):
            # Calculate function values
            f_x1 = self.evaluate_function(f, x_current)
            x2 = x_current + dx
            f_x2 = self.evaluate_function(f, x2)

            # Calculate product for sign change detection
            product = f_x1 * f_x2

            # Determine remark based on sign change
            if abs(f_x1) < tol:
                remark = "Root found"
                results.append((i + 1, x_current, dx, x2, f_x1, f_x2, product, remark))
                break
            elif product < 0:
                remark = "Sign change - Root exists between x₁ and x₂"
                results.append((i + 1, x_current, dx, x2, f_x1, f_x2, product, remark))
                # Reduce increment and continue from last x1
                dx = dx / 2
                continue
            else:
                remark = "Continue to next interval"

            # Store results
            results.append((i + 1, x_current, dx, x2, f_x1, f_x2, product, remark))

            # Move to next interval
            x_current = x2

            # Check if we've found a root
            if abs(f_x2) < tol:
                results.append((i + 2, x2, dx, x2 + dx, f_x2, self.evaluate_function(f, x2 + dx), 0, "Root found"))
                break

        return results

    def calculate(self):
        """Perform calculation and update display."""
        try:
            # Clear previous results
            self.tree.delete(*self.tree.get_children())
            self.ax.clear()

            # Get input values
            expr = self.parse_function(self.function.get())
            x1 = float(self.x1.get())
            dx = float(self.dx.get())
            max_iter = int(self.max_iter.get())
            tol = float(self.tol.get())

            # Perform Incremental Search iteration
            results = self.incremental_search(expr, x1, dx, max_iter, tol)

            # Update results table with color coding
            for result in results:
                values = [str(result[0])]  # Iteration number
                values.extend([f"{v:.6f}" if isinstance(v, float) else v for v in result[1:]])

                # Determine tag based on remark
                if "Root found" in result[7]:
                    tag = 'root_found'
                elif "Sign change" in result[7]:
                    tag = 'sign_change'
                else:
                    tag = 'continue'

                self.tree.insert('', 'end', values=values, tags=(tag,))

            # Enhanced plotting with dark theme
            x = Symbol('x')
            f = lambdify(x, expr, modules=['numpy'])

            # Determine plot range
            x_points = [r[1] for r in results] + [r[3] for r in results]
            x_min, x_max = min(x_points), max(x_points)
            x_range = x_max - x_min
            x_min -= 0.2 * x_range
            x_max += 0.2 * x_range

            # Create smooth curve
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

            # Plot iteration points with enhanced visibility
            for i, result in enumerate(results):
                x1_val, x2_val = result[1], result[3]
                f_x1, f_x2 = result[4], result[5]

                # Plot points with different colors based on result
                if "Root found" in result[7]:
                    color = self.colors["success"]
                    size = 10
                elif "Sign change" in result[7]:
                    color = self.colors["warning"]
                    size = 8
                else:
                    color = self.colors["interval"]
                    size = 6

                # Plot points
                self.ax.plot([x1_val], [f_x1], 'o', color=color, markersize=size, alpha=0.8)
                self.ax.plot([x2_val], [f_x2], 'o', color=color, markersize=size, alpha=0.8)

                # Plot increment lines
                self.ax.plot([x1_val, x2_val], [f_x1, f_x2], '--',
                             color=self.colors["midpoint"], alpha=0.5, linewidth=1)

                # Annotate important points
                if "Sign change" in result[7] or "Root found" in result[7]:
                    self.ax.annotate(f'Iter {result[0]}', (x1_val, f_x1),
                                     xytext=(10, 10), textcoords='offset points',
                                     color=self.colors["text_bright"], fontsize=9,
                                     bbox=dict(boxstyle="round,pad=0.3",
                                               facecolor=self.colors["bg_light"],
                                               edgecolor=color, alpha=0.8))

            # Set reasonable y-axis limits
            y_points = [y for y in y_vals if not (np.isinf(y) or np.isnan(y))]
            if y_points:
                y_min, y_max = min(y_points), max(y_points)
                y_range = y_max - y_min
                y_padding = 0.1 * y_range
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Enhanced plot styling
            self.ax.set_xlabel('x', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_ylabel('f(x)', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_title('Incremental Search Method Visualization',
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
                if "Root found" in final[7]:
                    root = final[1] if abs(final[4]) < tol else final[3]
                    messagebox.showinfo("Success",
                                        f"Root found: {root:.6f}\n"
                                        f"f(root): {self.evaluate_function(f, root):.6f}\n"
                                        f"Iterations: {len(results)}")
                else:
                    messagebox.showwarning("Warning",
                                           "Maximum iterations reached without finding a root.\n"
                                           "Try adjusting the initial value, increment, or maximum iterations.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear(self):
        """Clear all inputs and results."""
        self.function.delete(0, tk.END)
        self.x1.delete(0, tk.END)
        self.dx.delete(0, tk.END)
        self.max_iter.delete(0, tk.END)
        self.tol.delete(0, tk.END)
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()
        self.canvas.draw()

        # Reset default values
        self.function.insert(0, "x**2 - 4")
        self.x1.insert(0, "-3")
        self.dx.insert(0, "0.5")
        self.max_iter.insert(0, "50")
        self.tol.insert(0, "0.0001")

    def set_example(self, func, start_x, delta_x):
        """Set an example function with appropriate parameters."""
        self.function.delete(0, tk.END)
        self.function.insert(0, func)
        self.x1.delete(0, tk.END)
        self.x1.insert(0, start_x)
        self.dx.delete(0, tk.END)
        self.dx.insert(0, delta_x)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Incremental Search Method Calculator")

    # Set window size
    window_width = 1400
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app = IncrementalSearch(root)
    root.mainloop()
