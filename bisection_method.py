import tkinter as tk
from tkinter import ttk, messagebox
from sympy import symbols, sympify, lambdify, pi, E, exp, log, sqrt, Abs, sin, cos, tan, asin, acos, atan
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BisectionMethodApp:
    def __init__(self, master):
        self.master = master


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

        # Initialize interaction variables
        self.pan_active = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.zoom_factor = 1.1

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

        # Create results section with expanded layout
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
                               text="🔍 Bisection Method",
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

        self.function_entry = tk.Entry(input_frame,
                                       bg=self.colors["bg_light"],
                                       fg=self.colors["text_bright"],
                                       insertbackground=self.colors["accent"],
                                       font=("Segoe UI", 10),
                                       relief="flat", bd=5)
        self.function_entry.pack(fill=tk.X, pady=(0, 10))
        self.function_entry.insert(0, "x**3 - 2*x - 5")

        # Example functions
        tk.Label(input_frame, text="Examples:",
                 bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(5, 5))

        examples_frame = tk.Frame(input_frame, bg=self.colors["bg_dark"])
        examples_frame.pack(fill=tk.X)

        examples = [
            ("x³ - 2x - 5", "x**3 - 2*x - 5", "2", "3"),
            ("x² - 4", "x**2 - 4", "-3", "3"),
            ("sin(x) - x/2", "sin(x) - x/2", "0", "2"),
            ("e^x - 3x", "exp(x) - 3*x", "0", "2"),
            ("ln(x) - 1", "log(x) - 1", "1", "4"),
            ("cos(x) - x", "cos(x) - x", "0", "1")
        ]

        # Create two rows of example buttons
        for i, (label, func, a_val, b_val) in enumerate(examples[:3]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, a=a_val, b=b_val: self.set_example(f, a, b)).grid(
                row=0, column=i, padx=2, sticky="ew")

        for i, (label, func, a_val, b_val) in enumerate(examples[3:]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, a=a_val, b=b_val: self.set_example(f, a, b)).grid(
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
        tk.Label(grid_frame, text="Interval a:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.a_entry = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                fg=self.colors["text_bright"], width=10,
                                insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.a_entry.grid(row=0, column=1, sticky="ew", padx=(0, 15))
        self.a_entry.insert(0, "2")

        tk.Label(grid_frame, text="Interval b:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.b_entry = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                fg=self.colors["text_bright"], width=10,
                                insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.b_entry.grid(row=0, column=3, sticky="ew")
        self.b_entry.insert(0, "3")

        tk.Label(grid_frame, text="Tolerance:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        self.tol_entry = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                  fg=self.colors["text_bright"], width=10,
                                  insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.tol_entry.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=(10, 0))
        self.tol_entry.insert(0, "0.0001")

        tk.Label(grid_frame, text="Max Iterations:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=(10, 0))
        self.max_iter_entry = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                                       fg=self.colors["text_bright"], width=10,
                                       insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.max_iter_entry.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.max_iter_entry.insert(0, "50")

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
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Add reset view button
        self.reset_view_btn = tk.Button(buttons_frame, text="🔄 Reset View",
                                        bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                                        font=("Segoe UI", 10),
                                        relief="flat", bd=0, padx=20, pady=8,
                                        command=self.reset_plot_view)
        self.reset_view_btn.pack(side=tk.LEFT)

    def create_results_section(self):
        """Create expanded results section with table and plot"""
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
        columns = ('Iter', 'a', 'c', 'b', 'f(a)', 'f(c)', 'Error%', 'Sign', 'Next')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=15)

        # Configure columns
        column_widths = {'Iter': 50, 'a': 80, 'c': 80, 'b': 80,
                         'f(a)': 80, 'f(c)': 80, 'Error%': 70, 'Sign': 50, 'Next': 80}

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')

        # Configure row colors
        self.tree.tag_configure('first_subinterval',
                                background=self.colors["first_subinterval"],
                                foreground='white')
        self.tree.tag_configure('second_subinterval',
                                background=self.colors["second_subinterval"],
                                foreground='white')

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

        # Add interaction status
        self.interaction_status = tk.Label(plot_header, text="🖱️ Ready for interaction",
                                           bg=self.colors["bg_dark"], fg=self.colors["text_dim"],
                                           font=("Segoe UI", 9))
        self.interaction_status.pack(side=tk.RIGHT)

        # Create expanded matplotlib figure
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

        # Create legend frame
        self.legend_frame = tk.Frame(plot_frame, bg=self.colors["bg_dark"])
        self.legend_frame.pack(fill=tk.X, pady=(10, 0))

        # Create summary frame
        self.summary_frame = tk.Frame(plot_frame, bg=self.colors["bg_dark"])
        self.summary_frame.pack(fill=tk.X, pady=(10, 0))

        # Initialize empty legend and summary
        self.update_legend()
        self.update_summary()

    def setup_plot_interactions(self):
        """Setup mouse interactions for the plot"""
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

        # Update status
        self.interaction_status.config(text="🔍 Zooming...")
        self.master.update_idletasks()

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
            zoom_text = "🔍+ Zoom In"
        else:
            scale_factor = self.zoom_factor  # Zoom out
            zoom_text = "🔍- Zoom Out"

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

        # Update status
        self.interaction_status.config(text=f"{zoom_text} at ({xdata:.2f}, {ydata:.2f})")
        self.master.after(2000, lambda: self.interaction_status.config(text="🖱️ Ready for interaction"))

    def on_mouse_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left mouse button - start panning
            self.pan_active = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
            self.interaction_status.config(text="✋ Click and drag to pan")

        elif event.button == 3:  # Right mouse button - reset view
            self.reset_plot_view()

    def on_mouse_release(self, event):
        """Handle mouse button release"""
        if event.button == 1:  # Left mouse button
            self.pan_active = False
            self.pan_start_x = None
            self.pan_start_y = None
            self.interaction_status.config(text="🖱️ Ready for interaction")

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

        # Update status
        self.interaction_status.config(text=f"🔄 Panning... Δx={dx:.2f}, Δy={dy:.2f}")

    def reset_plot_view(self):
        """Reset plot to original view"""
        if self.original_xlim is not None and self.original_ylim is not None:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.canvas.draw_idle()
            self.interaction_status.config(text="🔄 View reset to original")
            self.master.after(2000, lambda: self.interaction_status.config(text="🖱️ Ready for interaction"))

    def update_legend(self, has_results=False):
        """Update legend with plot information"""
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        if not has_results:
            tk.Label(self.legend_frame, text="🎯 Click Calculate to see plot legend",
                     bg=self.colors["bg_dark"], fg=self.colors["text_dim"],
                     font=("Segoe UI", 10)).pack(pady=5)
            return

        # Create legend grid
        legend_grid = tk.Frame(self.legend_frame, bg=self.colors["bg_dark"])
        legend_grid.pack(fill=tk.X, padx=5)

        # Legend items
        legend_items = [
            ("━━", self.colors["function"], "Function f(x)"),
            ("●", self.colors["interval"], "Interval endpoints"),
            ("●", self.colors["midpoint"], "Midpoint"),
            ("┅┅", self.colors["grid"], "Zero line")
        ]

        for i, (symbol, color, text) in enumerate(legend_items):
            row = i // 2
            col = (i % 2) * 2

            # Color symbol
            tk.Label(legend_grid, text=symbol, bg=self.colors["bg_dark"],
                     fg=color, font=("Segoe UI", 12, "bold")).grid(
                row=row, column=col, padx=(5, 2), pady=2)

            # Text
            tk.Label(legend_grid, text=text, bg=self.colors["bg_dark"],
                     fg=self.colors["text"], font=("Segoe UI", 9)).grid(
                row=row, column=col + 1, sticky="w", padx=(0, 15), pady=2)

    def update_summary(self, summary_text=None):
        """Update summary with calculation results"""
        for widget in self.summary_frame.winfo_children():
            widget.destroy()

        if not summary_text:
            tk.Label(self.summary_frame, text="📋 Summary will appear after calculation",
                     bg=self.colors["bg_dark"], fg=self.colors["text_dim"],
                     font=("Segoe UI", 10)).pack(pady=5)
            return

        # Summary header
        tk.Label(self.summary_frame, text="📋 Calculation Summary:",
                 bg=self.colors["bg_dark"], fg=self.colors["accent"],
                 font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(5, 2))

        # Summary content
        tk.Label(self.summary_frame, text=summary_text,
                 bg=self.colors["bg_dark"], fg=self.colors["text"],
                 font=("Segoe UI", 10), wraplength=600, justify=tk.LEFT).pack(
            anchor=tk.W, padx=10)

    def set_example(self, func, a_val, b_val):
        """Set an example function with appropriate parameters."""
        self.function_entry.delete(0, tk.END)
        self.function_entry.insert(0, func)
        self.a_entry.delete(0, tk.END)
        self.a_entry.insert(0, a_val)
        self.b_entry.delete(0, tk.END)
        self.b_entry.insert(0, b_val)

    def parse_function(self, expr_str):
        """Parse function string to SymPy expression"""
        try:
            expr_str = expr_str.replace('^', '**')
            x = symbols('x')
            namespace = {
                'x': x, 'exp': exp, 'e': E, 'log': log, 'sqrt': sqrt,
                'abs': Abs, 'sin': sin, 'cos': cos, 'tan': tan,
                'asin': asin, 'acos': acos, 'atan': atan, 'pi': pi
            }
            return sympify(expr_str, locals=namespace)
        except Exception as e:
            raise ValueError(f"Error parsing function: {str(e)}")

    def evaluate_function(self, f, x_val):
        """Safely evaluate function at a point with enhanced error handling"""
        try:
            # Handle special cases
            if isinstance(x_val, complex):
                return None

            # Special handling for values very close to zero
            if abs(x_val) < 1e-15:
                # For sin(x)/x type functions, calculate the limit
                try:
                    h = 1e-10
                    left_val = float(f(x_val - h))
                    right_val = float(f(x_val + h))
                    if not (np.isnan(left_val) or np.isnan(right_val) or
                            np.isinf(left_val) or np.isinf(right_val)):
                        return (left_val + right_val) / 2
                except:
                    pass

            result = float(f(x_val))

            # Check for invalid results
            if np.isnan(result) or np.isinf(result):
                return None

            return result
        except Exception:
            return None

    def bisection_method(self, expr, a, b, tol, max_iter):
        """Perform bisection method calculation"""
        x = symbols('x')
        f = lambdify(x, expr, modules=['numpy'])
        results = []

        try:
            fa = self.evaluate_function(f, a)
            fb = self.evaluate_function(f, b)

            if fa * fb >= 0:
                raise ValueError("Function must have opposite signs at interval endpoints")

            prev_c = None
            for i in range(max_iter):
                c = (a + b) / 2
                fc = self.evaluate_function(f, c)

                # Calculate error percentage
                if prev_c is not None:
                    error = abs((c - prev_c) / c) * 100
                else:
                    error = None

                # Determine which subinterval to use
                product = fa * fc
                remark = "1st subinterval" if product < 0 else "2nd subinterval"

                results.append((i + 1, a, c, b, fa, fc, error, product, remark))

                # Check convergence
                if abs(fc) < tol or (error is not None and error < tol):
                    break

                prev_c = c
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc

            return results

        except Exception as e:
            raise ValueError(f"Error in bisection calculation: {str(e)}")

    def calculate(self):
        """Perform calculation and update display"""
        try:
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.ax.clear()

            # Get and validate inputs
            try:
                expr = self.parse_function(self.function_entry.get())
                a = float(self.a_entry.get())
                b = float(self.b_entry.get())
                tol = float(self.tol_entry.get())
                max_iter = int(self.max_iter_entry.get())

                if tol <= 0:
                    raise ValueError("Tolerance must be positive")
                if max_iter <= 0:
                    raise ValueError("Maximum iterations must be positive")
                if a >= b:
                    raise ValueError("Left endpoint must be less than right endpoint")

            except ValueError as e:
                messagebox.showerror("Input Error", str(e))
                return

            # Perform calculation
            results = self.bisection_method(expr, a, b, tol, max_iter)

            # Populate table with results
            for result in results:
                iter_num, xl, xc, xr, fl, fc, error, product, remark = result

                values = [
                    str(iter_num),
                    f"{xl:.6f}",
                    f"{xc:.6f}",
                    f"{xr:.6f}",
                    f"{fl:.6f}",
                    f"{fc:.6f}",
                    f"{error:.4f}" if error else "-",
                    "+" if product > 0 else "-",
                    remark
                ]

                tag = 'first_subinterval' if remark == "1st subinterval" else 'second_subinterval'
                self.tree.insert('', 'end', values=values, tags=(tag,))

            # Update plot
            self.update_plot(expr, results, a, b)

            # Update legend and summary
            self.update_legend(True)
            if results:
                final_result = results[-1]
                summary_text = (f"✅ Root found: x = {final_result[2]:.8f}\n"
                                f"🔄 Iterations: {len(results)}\n"
                                f"📊 Final error: {final_result[6]:.6f}%" if final_result[6] else "N/A")
                self.update_summary(summary_text)

        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")

    def update_plot(self, expr, results, orig_a, orig_b):
        """Update the expanded plot with function and iterations"""
        try:
            x = symbols('x')
            f = lambdify(x, expr, modules=['numpy'])

            # Determine plot range
            x_min, x_max = min(orig_a, orig_b), max(orig_a, orig_b)
            x_range = x_max - x_min
            x_min -= 0.5 * x_range
            x_max += 0.5 * x_range

            # Store original limits for reset functionality
            self.original_xlim = (x_min, x_max)

            # Plot function with improved handling
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = np.array([self.evaluate_function(f, x_val) for x_val in x_vals])

            # Replace None values with NaN for proper plotting
            y_vals = np.where(y_vals == None, np.nan, y_vals)

            # Plot the function as a continuous line where possible
            self.ax.plot(x_vals, y_vals, '-',
                         color=self.colors["function"],
                         linewidth=3,
                         label='f(x)',
                         alpha=0.9)

            # Calculate y limits and store for reset
            valid_y = y_vals[~np.isnan(y_vals)]
            if len(valid_y) > 0:
                y_margin = (np.max(valid_y) - np.min(valid_y)) * 0.1
                self.original_ylim = (np.min(valid_y) - y_margin, np.max(valid_y) + y_margin)
            else:
                self.original_ylim = (-10, 10)

            # Style plot
            self.ax.set_facecolor(self.colors["bg_dark"])
            self.ax.tick_params(colors=self.colors["text"], which='both', labelsize=11)

            # Plot zero line
            self.ax.axhline(y=0, color=self.colors["grid"],
                            linestyle='--', alpha=0.7, linewidth=2)

            # Plot iterations with enhanced markers
            for i, result in enumerate(results):
                a_val, c_val, b_val = result[1], result[2], result[3]
                fa, fc, fb = result[4], result[5], self.evaluate_function(f, b_val)

                # Plot points with larger markers
                self.ax.plot([a_val], [fa], 'o',
                             color=self.colors["interval"],
                             markersize=10, alpha=0.8,
                             markeredgecolor='white', markeredgewidth=2)
                self.ax.plot([b_val], [fb], 'o',
                             color=self.colors["interval"],
                             markersize=10, alpha=0.8,
                             markeredgecolor='white', markeredgewidth=2)
                self.ax.plot([c_val], [fc], 'o',
                             color=self.colors["midpoint"],
                             markersize=12, alpha=1.0,
                             markeredgecolor='white', markeredgewidth=2)

                # Annotate final result
                if i == len(results) - 1:
                    self.ax.annotate(f'🎯 Root ≈ {c_val:.6f}',
                                     (c_val, fc),
                                     xytext=(15, 25),
                                     textcoords='offset points',
                                     color=self.colors["text_bright"],
                                     fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor=self.colors["bg_light"],
                                               edgecolor=self.colors["accent"],
                                               alpha=0.9))

            # Enhanced grid and labels
            self.ax.grid(True, alpha=0.3, color=self.colors["grid"], linewidth=1)
            self.ax.set_xlabel('x', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_ylabel('f(x)', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_title('Interactive Bisection Method Visualization',
                              color=self.colors["text_bright"], fontsize=14, fontweight='bold', pad=20)

            # Style spines
            for spine in self.ax.spines.values():
                spine.set_color(self.colors["grid"])
                spine.set_linewidth(1.5)

            # Set initial view
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)

            self.fig.tight_layout()
            self.canvas.draw()

            # Update interaction status
            self.interaction_status.config(text="🖱️ Plot ready - Scroll to zoom, drag to pan")

        except Exception as e:
            print(f"Plot update failed: {e}")

    def clear(self):
        """Clear all inputs and results"""
        # Clear inputs
        self.function_entry.delete(0, tk.END)
        self.a_entry.delete(0, tk.END)
        self.b_entry.delete(0, tk.END)
        self.tol_entry.delete(0, tk.END)
        self.max_iter_entry.delete(0, tk.END)

        # Clear results
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax.clear()
        self.canvas.draw()

        # Reset defaults
        self.function_entry.insert(0, "x**3 - 2*x - 5")
        self.a_entry.insert(0, "2")
        self.b_entry.insert(0, "3")
        self.tol_entry.insert(0, "0.0001")
        self.max_iter_entry.insert(0, "50")

        # Reset legend and summary
        self.update_legend(False)
        self.update_summary()

        # Reset interaction status
        self.interaction_status.config(text="🖱️ Ready for interaction")

        # Reset original limits
        self.original_xlim = None
        self.original_ylim = None


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bisection Method")

    # Set expanded window size
    window_width = 1400
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app = BisectionMethodApp(root)
    root.mainloop()