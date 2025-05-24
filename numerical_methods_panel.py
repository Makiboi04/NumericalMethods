import tkinter as tk
from tkinter import ttk, messagebox
import sys

# Import the numerical method classes
try:
    from bisection_method import BisectionMethodApp
    from false_position import FalsePosition
    from newton_raphson_method import NewtonRaphson
    from secant_method import SecantMethod
    from incremental_search import IncrementalSearch
except ImportError as e:
    print(f"Error importing numerical method classes: {e}")
    print("Please ensure all numerical method files are in the same directory.")
    sys.exit(1)


class ThemeManager:
    """Centralized theme manager to prevent conflicts"""
    _theme_created = False
    _style = None

    @classmethod
    def setup_global_theme(cls):
        """Setup global dark theme once"""
        if cls._theme_created:
            return cls._style

        try:
            cls._style = ttk.Style()

            theme_name = "global_dark_theme"

            if theme_name not in cls._style.theme_names():
                cls._style.theme_create(theme_name, parent="alt", settings={
                    "Treeview": {
                        "configure": {
                            "background": "#1a1b26",
                            "foreground": "#c0caf5",
                            "fieldbackground": "#1a1b26",
                            "borderwidth": 0
                        }
                    },
                    "Treeview.Heading": {
                        "configure": {
                            "background": "#24283b",
                            "foreground": "#7aa2f7",
                            "borderwidth": 0,
                            "relief": "flat"
                        }
                    },
                    "Scrollbar": {
                        "configure": {
                            "background": "#24283b",
                            "troughcolor": "#1a1b26",
                            "borderwidth": 0
                        }
                    }
                })

            cls._style.theme_use(theme_name)
            cls._theme_created = True
            print(f"Global theme '{theme_name}' created successfully")

        except Exception as e:
            print(f"Warning: Could not setup global theme: {e}")

        return cls._style


class MethodWrapper:
    """Wrapper class to adapt method classes for dashboard integration"""

    def __init__(self, parent_frame, method_class, method_name):
        self.parent_frame = parent_frame
        self.method_name = method_name

        self.container_frame = tk.Frame(parent_frame, bg="#1a1b26")
        self.container_frame.pack(fill=tk.BOTH, expand=True)

        # Create the method instance with theme management
        self.method_instance = self.create_method_instance(method_class)

    def create_method_instance(self, method_class):
        """Create method instance with frame adaptation and theme management"""

        # Create a mock master that has the title attribute but uses our frame
        class FrameWithTitle:
            def __init__(self, frame):
                self.frame = frame
                self._title = ""

            def title(self, title_text=None):
                if title_text is not None:
                    self._title = title_text
                return self._title

            def configure(self, **kwargs):
                return self.frame.configure(**kwargs)

            def pack(self, **kwargs):
                return self.frame.pack(**kwargs)

            def grid(self, **kwargs):
                return self.frame.grid(**kwargs)

            def place(self, **kwargs):
                return self.frame.place(**kwargs)

            def pack_propagate(self, flag):
                return self.frame.pack_propagate(flag)

            def grid_columnconfigure(self, *args, **kwargs):
                return self.frame.grid_columnconfigure(*args, **kwargs)

            def grid_rowconfigure(self, *args, **kwargs):
                return self.frame.grid_rowconfigure(*args, **kwargs)

            def winfo_screenwidth(self):
                return self.frame.winfo_screenwidth()

            def winfo_screenheight(self):
                return self.frame.winfo_screenheight()

            def geometry(self, *args):
                # Ignore geometry calls since we're embedded
                pass

            def __getattr__(self, name):
                # Delegate any other attributes to the frame
                return getattr(self.frame, name)

        # Create the mock master
        mock_master = FrameWithTitle(self.container_frame)

        # Temporarily patch the method's theme creation to prevent conflicts
        original_configure_ttk_style = None
        if hasattr(method_class, 'configure_ttk_style'):
            # Store original method
            original_configure_ttk_style = getattr(method_class, 'configure_ttk_style')

            # Create a no-op replacement
            def dummy_configure_ttk_style(self):
                """Dummy method to prevent theme conflicts"""
                # Use the global theme instead
                self.style = ThemeManager.setup_global_theme()
                print(f"Using global theme for {self.__class__.__name__}")

            # Temporarily replace the method
            setattr(method_class, 'configure_ttk_style', dummy_configure_ttk_style)

        try:
            # Create the method instance
            method_instance = method_class(mock_master)

            # If the method has a style attribute, make sure it uses our global theme
            if hasattr(method_instance, 'style'):
                method_instance.style = ThemeManager.setup_global_theme()

            return method_instance

        finally:
            # Restore original method if we patched it
            if original_configure_ttk_style:
                setattr(method_class, 'configure_ttk_style', original_configure_ttk_style)

    def destroy(self):
        """Clean up the wrapper"""
        if hasattr(self.method_instance, 'master'):
            try:
                # Don't destroy the master since it's our frame
                pass
            except:
                pass
        self.container_frame.destroy()


class NumericalMethodsDashboard:
    def __init__(self, master):
        self.master = master
        self.master.title("Numerical Methods Dashboard")

        # Enhanced color scheme
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
            "sidebar_bg": "#16161e",
            "button_hover": "#3d59a1",
            "button_selected": "#2ac3de"
        }

        # Configure root window
        self.master.configure(bg=self.colors["bg_dark"])

        # Set window size
        window_width = 1400
        window_height = 900
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.master.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # Setup global theme first
        self.setup_global_theme()

        # Create main container with grid layout
        self.master.grid_columnconfigure(0, weight=0)  # Sidebar column (fixed width)
        self.master.grid_columnconfigure(1, weight=1)  # Content column (expandable)
        self.master.grid_rowconfigure(0, weight=1)  # Make the row expandable

        # Create sidebar frame
        self.create_sidebar()

        # Create content frame
        self.content_frame = tk.Frame(self.master, bg=self.colors["bg_dark"])
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        # Initialize method tracking
        self.current_method_wrapper = None

        # Show welcome screen initially
        self.show_welcome_screen()

    def setup_global_theme(self):
        """Setup the global dark theme"""
        self.style = ThemeManager.setup_global_theme()

    def create_sidebar(self):
        """Create sidebar with method buttons"""
        sidebar_width = 250

        # Create sidebar frame
        self.sidebar = tk.Frame(self.master, bg=self.colors["sidebar_bg"], width=sidebar_width)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.pack_propagate(False)

        # Create header
        header_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar_bg"])
        header_frame.pack(fill=tk.X, pady=(20, 30))

        title_label = tk.Label(header_frame,
                               text="🧮 Numerical Methods",
                               bg=self.colors["sidebar_bg"],
                               fg=self.colors["accent"],
                               font=("Segoe UI", 16, "bold"))
        title_label.pack(anchor=tk.CENTER)

        subtitle_label = tk.Label(header_frame,
                                  text="Root Finding Techniques",
                                  bg=self.colors["sidebar_bg"],
                                  fg=self.colors["text_dim"],
                                  font=("Segoe UI", 10))
        subtitle_label.pack(anchor=tk.CENTER, pady=(5, 0))

        # Add separator line
        separator = tk.Frame(header_frame, bg=self.colors["accent"], height=2)
        separator.pack(fill=tk.X, pady=(15, 0))

        # Create method buttons
        self.create_method_buttons()

        # Add footer
        footer_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar_bg"])
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        footer_label = tk.Label(footer_frame,
                                text="Select a method to begin",
                                bg=self.colors["sidebar_bg"],
                                fg=self.colors["text_dim"],
                                font=("Segoe UI", 9))
        footer_label.pack(anchor=tk.CENTER)

    def create_method_buttons(self):
        """Create buttons for each numerical method"""
        methods = [
            {
                "id": "bisection",
                "name": "Bisection Method",
                "icon": "🔍",
                "description": "Bracket-based root finding",
                "class": BisectionMethodApp
            },
            {
                "id": "false_position",
                "name": "False Position",
                "icon": "📊",
                "description": "Linear interpolation method",
                "class": FalsePosition
            },
            {
                "id": "newton_raphson",
                "name": "Newton-Raphson",
                "icon": "📈",
                "description": "Derivative-based method",
                "class": NewtonRaphson
            },
            {
                "id": "secant",
                "name": "Secant Method",
                "icon": "📉",
                "description": "Derivative-free method",
                "class": SecantMethod
            },
            {
                "id": "incremental_search",
                "name": "Incremental Search",
                "icon": "🔎",
                "description": "Systematic interval search",
                "class": IncrementalSearch
            }
        ]

        self.selected_button = None
        self.buttons = {}
        self.method_classes = {method["id"]: method["class"] for method in methods}

        # Create a frame for the buttons
        buttons_frame = tk.Frame(self.sidebar, bg=self.colors["sidebar_bg"])
        buttons_frame.pack(fill=tk.X, padx=15, pady=(0, 20))

        # Create each button
        for method in methods:
            button_container = tk.Frame(buttons_frame, bg=self.colors["sidebar_bg"])
            button_container.pack(fill=tk.X, pady=8)

            button = tk.Button(
                button_container,
                text=f"{method['icon']} {method['name']}",
                bg=self.colors["bg_light"],
                fg=self.colors["text_bright"],
                font=("Segoe UI", 11, "bold"),
                relief="flat",
                bd=0,
                padx=20,
                pady=15,
                anchor="w",
                width=28,
                command=lambda m=method: self.show_method(m["id"], m["class"], m["name"])
            )
            button.pack(fill=tk.X)

            desc_label = tk.Label(
                button_container,
                text=method['description'],
                bg=self.colors["sidebar_bg"],
                fg=self.colors["text_dim"],
                font=("Segoe UI", 8),
                anchor="w"
            )
            desc_label.pack(fill=tk.X, padx=20, pady=(2, 0))

            button.bind("<Enter>", lambda e, b=button: self.on_button_hover(b))
            button.bind("<Leave>", lambda e, b=button: self.on_button_leave(b))

            self.buttons[method["id"]] = button

    def on_button_hover(self, button):
        if button != self.selected_button:
            button.config(bg=self.colors["button_hover"])

    def on_button_leave(self, button):
        if button != self.selected_button:
            button.config(bg=self.colors["bg_light"])

    def show_welcome_screen(self):
        """Show welcome screen when no method is selected"""
        self.clear_content_frame()

        welcome_frame = tk.Frame(self.content_frame, bg=self.colors["bg_dark"])
        welcome_frame.pack(fill=tk.BOTH, expand=True)

        center_frame = tk.Frame(welcome_frame, bg=self.colors["bg_dark"])
        center_frame.place(relx=0.5, rely=0.5, anchor="center")

        title_label = tk.Label(center_frame,
                               text="🧮 Welcome to Numerical Methods Dashboard",
                               bg=self.colors["bg_dark"],
                               fg=self.colors["accent"],
                               font=("Segoe UI", 24, "bold"))
        title_label.pack(pady=(0, 20))

        desc_text = """Explore various numerical methods for finding roots of equations.

Select a method from the sidebar to get started:

🔍 Bisection Method - Reliable bracket-based approach
📊 False Position - Faster convergence with linear interpolation  
📈 Newton-Raphson - Rapid convergence using derivatives
📉 Secant Method - Derivative-free alternative to Newton-Raphson
🔎 Incremental Search - Systematic interval searching

Each method includes interactive visualizations and detailed iteration tables."""

        desc_label = tk.Label(center_frame,
                              text=desc_text,
                              bg=self.colors["bg_dark"],
                              fg=self.colors["text"],
                              font=("Segoe UI", 12),
                              justify=tk.CENTER,
                              wraplength=600)
        desc_label.pack(pady=(0, 30))

    def clear_content_frame(self):
        """Clear the content frame and clean up current method"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        if self.current_method_wrapper:
            try:
                self.current_method_wrapper.destroy()
            except:
                pass
            self.current_method_wrapper = None

    def show_method(self, method_id, method_class, method_name):
        """Show the selected method"""
        print(f"Loading {method_name}...")

        # Update button styles
        if self.selected_button:
            self.selected_button.config(bg=self.colors["bg_light"])

        self.selected_button = self.buttons[method_id]
        self.selected_button.config(bg=self.colors["button_selected"])

        # Clear content frame
        self.clear_content_frame()

        # Create new method wrapper
        try:
            self.current_method_wrapper = MethodWrapper(
                self.content_frame,
                method_class,
                method_name
            )
            print(f"Successfully loaded {method_name}")

        except Exception as e:
            error_msg = f"Failed to load {method_name}: {str(e)}"
            print(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.show_welcome_screen()

    def on_closing(self):
        """Handle application closing"""
        self.clear_content_frame()
        self.master.destroy()


def main():
    """Main function to run the dashboard"""
    root = tk.Tk()

    # Create dashboard instance
    dashboard = NumericalMethodsDashboard(root)

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", dashboard.on_closing)

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()