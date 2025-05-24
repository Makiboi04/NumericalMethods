import numpy as np
from sympy import (symbols, sympify, lambdify, E, exp, Symbol, I,
                   log, sqrt, Abs, sin, cos, tan, asin, acos, atan,
                   csc, sec, cot, acsc, asec, acot,
                   sinh, cosh, tanh, asinh, acosh, atanh,
                   pi, factorial, gamma, ceiling, floor, sign)
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re

class FalsePosition:
    def __init__(self, master):
        self.master = master
        self.master.title("False Position Method")


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
                               text="🔍 False Position Method",
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
        self.function.insert(0, "x**3 - 2*x - 5")

        # Example functions
        tk.Label(input_frame, text="Examples:",
                 bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                 font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(5, 5))

        examples_frame = tk.Frame(input_frame, bg=self.colors["bg_dark"])
        examples_frame.pack(fill=tk.X)

        examples = [
            ("x² - 4", "x**2 - 4", "-3", "0"),
            ("sin(x) - x/2", "sin(x) - x/2", "1", "2"),
            ("x³ - 2x - 5", "x**3 - 2*x - 5", "2", "3"),
            ("e^x - 3x", "exp(x) - 3*x", "1", "2"),
            ("ln(x²+1) - 2", "log(x**2 + 1) - 2", "1", "2"),
            ("cos(x) - x·e^(-x)", "cos(x) - x*exp(-x)", "0", "1")
        ]

        # Create two rows of example buttons
        for i, (label, func, _, _) in enumerate(examples[:3]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, ex=examples[i]: self.set_example(f, ex[2], ex[3])).grid(row=0, column=i,
                                                                                                     padx=2,
                                                                                                     sticky="ew")
        for i, (label, func, _, _) in enumerate(examples[3:]):
            tk.Button(examples_frame, text=label,
                      bg=self.colors["bg_light"], fg=self.colors["text_bright"],
                      font=("Segoe UI", 8), relief="flat", bd=0, padx=10, pady=4,
                      command=lambda f=func, ex=examples[i + 3]: self.set_example(f, ex[2], ex[3])).grid(row=1,
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
        tk.Label(grid_frame, text="Lower bound xL:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.xL = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.xL.grid(row=0, column=1, sticky="ew", padx=(0, 15))
        self.xL.insert(0, "2")

        tk.Label(grid_frame, text="Upper bound xU:", bg=self.colors["bg_dark"],
                 fg=self.colors["text"]).grid(row=0, column=2, sticky="w", padx=(0, 5))
        self.xU = tk.Entry(grid_frame, bg=self.colors["bg_light"],
                           fg=self.colors["text_bright"], width=10,
                           insertbackground=self.colors["accent"], relief="flat", bd=3)
        self.xU.grid(row=0, column=3, sticky="ew")
        self.xU.insert(0, "3")

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

        # Illinois modification checkbox
        self.use_illinois = tk.BooleanVar(value=True)
        illinois_check = tk.Checkbutton(buttons_frame, text="Use Illinois Modification",
                                        variable=self.use_illinois,
                                        bg=self.colors["bg_dark"], fg=self.colors["text_bright"],
                                        selectcolor=self.colors["bg_light"],
                                        activebackground=self.colors["bg_dark"],
                                        activeforeground=self.colors["text_bright"],
                                        font=("Segoe UI", 10))
        illinois_check.pack(side=tk.LEFT, padx=(0, 20))

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
        columns = ('n', 'xL', 'xU', 'xR', 'f(xL)', 'f(xU)', 'f(xR)', 'Error')
        self.tree = ttk.Treeview(table_container, columns=columns, show='headings', height=15)

        # Configure columns
        column_widths = {'n': 40, 'xL': 80, 'xU': 80, 'xR': 80,
                         'f(xL)': 80, 'f(xU)': 80, 'f(xR)': 80, 'Error': 80}

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths[col], anchor='center')

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

        # Store original limits for reset functionality
        self.original_xlim = None
        self.original_ylim = None

    def setup_plot_interactions(self):
        """Setup mouse interactions for the plot"""
        # Connect mouse events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

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

    def preprocess_function(self, expr_str):
        """Preprocess the function string to handle various mathematical notations."""
        # Replace common alternative notations
        replacements = {
            'sin⁻¹': 'asin',
            'cos⁻¹': 'acos',
            'tan⁻¹': 'atan',
            'cosec': 'csc',
            'arcsin': 'asin',
            'arccos': 'acos',
            'arctan': 'atan',
            'arccsc': 'acsc',
            'arcsec': 'asec',
            'arccot': 'acot',
            'arcsinh': 'asinh',
            'arccosh': 'acosh',
            'arctanh': 'atanh',
            'ln': 'log',
            'log₁₀': 'log10',
            'log₂': 'log2',
            '²': '**2',
            '³': '**3',
            '⁴': '**4',
            '₁': '1',
            '₂': '2',
            '₃': '3',
            '₄': '4',
            'π': 'pi',
            '√': 'sqrt',
            '∛': 'cbrt',
            '∜': 'root4',
            '|': 'abs(',
            '!': 'factorial(',
            '⌈': 'ceiling(',
            '⌊': 'floor(',
            'sgn': 'sign'
        }

        for old, new in replacements.items():
            expr_str = expr_str.replace(old, new)

        # Handle implicit multiplication (e.g., 2x -> 2*x)
        expr_str = re.sub(r'(\d+)([a-zA-Z\(])', r'\1*\2', expr_str)
        expr_str = re.sub(r'(\))([a-zA-Z\(])', r'\1*\2', expr_str)

        # Handle special functions
        expr_str = re.sub(r'cbrt$$(.*?)$$', r'((\1)**(1/3))', expr_str)
        expr_str = re.sub(r'root4$$(.*?)$$', r'((\1)**(1/4))', expr_str)
        expr_str = re.sub(r'log2$$(.*?)$$', r'(log(\1)/log(2))', expr_str)
        expr_str = re.sub(r'log10$$(.*?)$$', r'(log(\1)/log(10))', expr_str)

        # Handle exponentials
        expr_str = re.sub(r'e\^$$([^)]+)$$', r'exp(\1)', expr_str)
        expr_str = re.sub(r'e\^([a-zA-Z0-9_.]+)', r'exp(\1)', expr_str)
        expr_str = expr_str.replace('^', '**')

        return expr_str

    def parse_function(self, expr_str):
        """Parse the function string to a SymPy expression with enhanced support for complex functions."""
        try:
            # Preprocess the expression
            expr_str = self.preprocess_function(expr_str)

            x = Symbol('x')
            # Extended namespace with more mathematical functions
            namespace = {
                'x': x,
                'exp': exp,
                'e': E,
                'E': E,
                'pi': pi,
                'i': I,
                'I': I,
                'log': log,
                'sqrt': sqrt,
                'abs': Abs,
                # Trigonometric functions
                'sin': sin,
                'cos': cos,
                'tan': tan,
                'csc': csc,
                'sec': sec,
                'cot': cot,
                # Inverse trigonometric functions
                'asin': asin,
                'acos': acos,
                'atan': atan,
                'acsc': acsc,
                'asec': asec,
                'acot': acot,
                # Hyperbolic functions
                'sinh': sinh,
                'cosh': cosh,
                'tanh': tanh,
                'asinh': asinh,
                'acosh': acosh,
                'atanh': atanh,
                # Special functions
                'factorial': factorial,
                'gamma': gamma,
                'ceil': ceiling,
                'floor': floor,
                'sign': sign
            }

            return sympify(expr_str, locals=namespace)
        except Exception as e:
            raise ValueError(f"Error parsing function: {str(e)}")

    def evaluate_function(self, f, x_val):
        """Safely evaluate a function at a point with enhanced error handling."""
        try:
            # Handle potential complex results
            result = complex(f(x_val))

            # If result is essentially real (imaginary part very small)
            if abs(result.imag) < 1e-10:
                return float(result.real)

            # For complex results, return magnitude (for root finding)
            # This allows finding roots even when function passes through complex plane
            return abs(result)

        except Exception as e:
            # Handle specific math errors
            if isinstance(e, (ValueError, ZeroDivisionError, OverflowError)):
                # Return a very large value to indicate invalid region
                return float('inf')
            raise ValueError(f"Error evaluating function at x={x_val}: {str(e)}")

    def regula_falsi(self, expr, xL, xU, tol, max_iter, use_illinois=True):
        """
        Perform False Position (Regula Falsi) iteration.

        Parameters:
        -----------
        expr : sympy expression
            The function to find roots for
        xL, xU : float
            Lower and upper bounds of the interval
        tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations
        use_illinois : bool
            Whether to use Illinois modification

        Returns:
        --------
        list of tuples
            Each tuple contains (iteration, xL, xU, xR, f(xL), f(xU), f(xR), error)
        """
        x = Symbol('x')
        f = lambdify(x, expr, modules=['numpy'])
        results = []

        # Initial function evaluations
        f_xL = self.evaluate_function(f, xL)
        f_xU = self.evaluate_function(f, xU)

        # Check if initial bracket is valid
        if f_xL * f_xU >= 0:
            raise ValueError("Initial bracket does not contain a root (f(xL) and f(xU) must have opposite signs)")

        # Initialize variables
        xL_current = xL
        xU_current = xU
        f_xL_current = f_xL
        f_xU_current = f_xU
        xR_prev = None

        for i in range(max_iter):
            # Calculate the false position point
            try:
                denominator = f_xU_current - f_xL_current
                if abs(denominator) < 1e-15:  # Avoid division by very small numbers
                    raise ValueError("Method failed: denominator too close to zero")

                xR = xU_current - f_xU_current * (xU_current - xL_current) / denominator
                f_xR = self.evaluate_function(f, xR)
            except Exception as e:
                raise ValueError(f"Error in iteration {i + 1}: {str(e)}")

            # Calculate relative error if possible
            if xR_prev is not None and abs(xR) > 1e-15:
                error = abs((xR - xR_prev) / xR)
            else:
                error = float('inf')

            # Store current iteration results
            results.append((i + 1, xL_current, xU_current, xR, f_xL_current, f_xU_current, f_xR, error))

            # Check for convergence
            if abs(f_xR) < tol or (xR_prev is not None and error < tol):
                break

            # Update the bracket
            if f_xR * f_xL_current < 0:  # Root is in left half
                xU_current = xR
                f_xU_current = f_xR
                if use_illinois and abs(f_xL_current) > abs(f_xR):
                    f_xL_current /= 2  # Illinois modification
            else:  # Root is in right half
                xL_current = xR
                f_xL_current = f_xR
                if use_illinois and abs(f_xU_current) > abs(f_xR):
                    f_xU_current /= 2  # Illinois modification

            xR_prev = xR

            # Additional convergence check
            if abs(xU_current - xL_current) < tol:
                break

        return results

    def calculate(self):
        """Perform calculation and update display with enhanced plotting."""
        try:
            # Clear previous results
            self.tree.delete(*self.tree.get_children())
            self.ax.clear()

            # Get input values
            expr = self.parse_function(self.function.get())
            xL = float(self.xL.get())
            xU = float(self.xU.get())
            tol = float(self.tol.get())
            max_iter = int(self.max_iter.get())
            use_illinois = self.use_illinois.get()

            # Perform Regula Falsi iteration
            results = self.regula_falsi(expr, xL, xU, tol, max_iter, use_illinois)

            # Update results table
            for result in results:
                self.tree.insert('', 'end', values=[f"{v:.6f}" if isinstance(v, float) else v for v in result])

            # Enhanced plotting with dark theme
            x = Symbol('x')
            f = lambdify(x, expr, modules=['numpy', 'sympy'])

            # Determine plot range with more points for smooth curves
            x_points = [r[1] for r in results] + [r[2] for r in results] + [r[3] for r in results]
            x_min, x_max = min(x_points), max(x_points)
            x_range = x_max - x_min
            x_min -= 0.2 * x_range
            x_max += 0.2 * x_range

            # Store original limits for reset functionality
            self.original_xlim = (x_min, x_max)

            # Create smooth curve with more points
            x_vals = np.linspace(x_min, x_max, 500)
            y_vals = [self.evaluate_function(f, x) for x in x_vals]

            # Remove infinity and NaN values for plotting
            valid_points = [(x, y) for x, y in zip(x_vals, y_vals)
                            if not (np.isinf(y) or np.isnan(y))]
            if valid_points:
                x_vals, y_vals = zip(*valid_points)

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
            x_roots = [r[3] for r in results]
            y_roots = [r[6] for r in results]
            self.ax.plot(x_roots, y_roots, 'o-', color=self.colors["interval"],
                         label='Iterations', markersize=6, alpha=0.8, linewidth=2)

            # Highlight final root point
            if results:
                final = results[-1]
                self.ax.plot(final[3], final[6], 'o', color=self.colors["midpoint"],
                             label='Final Root', markersize=12, alpha=1.0,
                             markeredgecolor='white', markeredgewidth=2)

                # Annotate final result
                self.ax.annotate(f'🎯 Root ≈ {final[3]:.6f}',
                                 (final[3], final[6]),
                                 xytext=(15, 25),
                                 textcoords='offset points',
                                 color=self.colors["text_bright"],
                                 fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.5",
                                           facecolor=self.colors["bg_light"],
                                           edgecolor=self.colors["accent"],
                                           alpha=0.9))

            # Set reasonable y-axis limits with padding and store for reset
            y_points = [y for y in y_vals if not (np.isinf(y) or np.isnan(y))]
            if y_points:
                y_min, y_max = min(y_points), max(y_points)
                y_range = y_max - y_min
                y_padding = 0.1 * y_range
                self.original_ylim = (y_min - y_padding, y_max + y_padding)
                self.ax.set_ylim(self.original_ylim)
            else:
                self.original_ylim = (-10, 10)
                self.ax.set_ylim(self.original_ylim)

            # Enhanced plot styling
            self.ax.set_xlabel('x', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_ylabel('f(x)', color=self.colors["text_bright"], fontsize=12, fontweight='bold')
            self.ax.set_title('Interactive False Position Method Visualization',
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

            # Set initial view
            self.ax.set_xlim(self.original_xlim)

            self.fig.tight_layout()
            self.canvas.draw()

            # Update interaction status
            self.interaction_status.config(text="🖱️ Plot ready - Scroll to zoom, drag to pan")

            # Show final result with more details
            if results:
                final = results[-1]
                messagebox.showinfo("Success",
                                    f"Root found: {final[3]:.8f}\n"
                                    f"f(root): {final[6]:.8f}\n"
                                    f"Error: {final[7]:.8f}\n"
                                    f"Iterations: {len(results)}\n"
                                    f"Illinois Modification: {'Yes' if use_illinois else 'No'}\n"
                                    f"Convergence achieved: {'Yes' if final[7] < tol else 'No'}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear(self):
        """Clear all inputs and results."""
        self.function.delete(0, tk.END)
        self.xL.delete(0, tk.END)
        self.xU.delete(0, tk.END)
        self.tol.delete(0, tk.END)
        self.max_iter.delete(0, tk.END)
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()
        self.canvas.draw()

        # Reset default values
        self.function.insert(0, "x**3 - 2*x - 5")
        self.xL.insert(0, "2")
        self.xU.insert(0, "3")
        self.tol.insert(0, "0.0001")
        self.max_iter.insert(0, "50")

        # Reset interaction status
        self.interaction_status.config(text="🖱️ Ready for interaction")

        # Reset original limits
        self.original_xlim = None
        self.original_ylim = None

    def set_example(self, func, xL, xU):
        """Set an example function with appropriate bounds."""
        self.function.delete(0, tk.END)
        self.function.insert(0, func)
        self.xL.delete(0, tk.END)
        self.xL.insert(0, xL)
        self.xU.delete(0, tk.END)
        self.xU.insert(0, xU)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("False Position Method")

    # Set window size
    window_width = 1400
    window_height = 900
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    app = FalsePosition(root)
    root.mainloop()