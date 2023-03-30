from aiaccel.command_line_options import CommandLineOptions


class TestCommandLineOptions:
    def test_init(self) -> None:
        options = CommandLineOptions(
            config="config.yaml",
            resume=0,
            clean=True,
            process_name="test"
        )
        assert options.config == "config.yaml"
        assert options.resume == 0
        assert options.clean is True
        assert options.process_name == "test"
