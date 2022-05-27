class Terminal:
    """The any color text output to Terminal.

    Attributes:
        colors (dict): set of available colors
    """
    def __init__(self):
        self.colors = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "lightgray": "\033[37m",
            "default": "\033[39m",
            "darkgray": "\033[90m",
            "lightred": "\033[91m",
            "lightgreen": "\033[92m",
            "lightyellow": "\033[93m",
            "lightblue": "\033[94m",
            "lightmagenta": "\033[95m",
            "lightcyan": "\033[96m",
            "white": "\033[97m",
            "reset": "\033[0m",
        }

    def set_terminal_color(self, color):
        """ Set text color.
        Args:
            color (str): Any colors.

        Returns:
            None

        Note:
            The color of the text will change until self.reset() is called.
        """
        print("{}".format(self.colors[color]), end="")

    def reset(self):
        """ Reset the default text color of the terminal.

        Args:
            None

        Returns:
            None
        """
        print("{}".format(self.colors["reset"]), end="")

    def print_note(self, message):
        """ Print when notification(note).

        Args:
            None

        Returns:
            None
        """
        self.set_terminal_color("green")
        print("[NOTE] {}".format(message))
        self.reset()

    def print_warning(self, message):
        """ Print when warning.

        Args:
            None

        Returns:
            None
        """
        self.set_terminal_color("yellow")
        print("[WARNING] {}".format(message))
        self.reset()

    def print_error(self, message):
        """ Print when error.

        Args:
            None

        Returns:
            None
        """
        self.set_terminal_color("red")
        print("[ERROR] {}".format(message))
        self.reset()

    def cprint(self, message: str, color: str):
        """ Set any color to any text.

        Args:
            None

        Returns:
            None
        """
        if color in self.colors.keys():
            self.set_terminal_color(color)
            print(message)
            self.reset()
        else:
            self.set_terminal_color("red")
            print(message)
            self.reset()
