"""Subprocess coverage of the root module entry point, without AWS access."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServeModuleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="serve module ")
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()
        shutil.copyfile(ROOT / "serve.py", self.project / "serve.py")
        shutil.copytree(ROOT / "src", self.project / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(ROOT / "deployment_config.example.json",
                        self.project / "deployment_config.json")
        # Fail loudly if argument handling ever reaches an AWS import or child.
        (self.project / "src" / "boto3.py").write_text(
            "raise AssertionError('AWS import during argument handling')\n")
        (self.project / "src" / "subprocess.py").write_text(
            "def run(*args, **kwargs):\n"
            "    raise AssertionError('child process during argument handling')\n")

    def run_cli(self, *arguments):
        env = {key: value for key, value in os.environ.items()
               if key not in {"PYTHONPATH", "PYTHONHOME"}}
        return subprocess.run(
            [sys.executable, "-S", "-m", "serve", *arguments],
            cwd=self.project, env=env, capture_output=True, text=True, timeout=15)

    def test_help_without_config_or_site_packages(self):
        (self.project / "deployment_config.json").unlink()
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALIAS", result.stdout)
        self.assertIn("--ttl-seconds", result.stdout)

    def test_required_and_invalid_alias_without_side_effects(self):
        for arguments, message in [
            ([], "required"),
            (["unknown"], "unknown alias"),
            (["Qwen3-27B-Uncensored-Q6", "--ttl-seconds", "0"], "positive"),
        ]:
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertNotIn("AssertionError", result.stderr)
                if arguments == ["unknown"]:
                    self.assertIn(str(self.project / "deployment_config.json"), result.stderr)

    def test_forwards_arguments_source_location_and_exit_status(self):
        # Replace only the implementation in this disposable project. The real
        # shim must execute it as a script with sibling imports and intact argv.
        (self.project / "src" / "serve.py").write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "from deployment import parse_args\n"
            "if __name__ == '__main__':\n"
            "    args = parse_args('fixture', [sys.argv[1]])\n"
            "    print(json.dumps([sys.argv[1:], str(Path(__file__).resolve()), args.config['name']]))\n"
            "    sys.exit(int(sys.argv[-1]))\n")
        for status in (0, 7):
            arguments = ["Qwen3-27B-Uncensored-Q6", "--ttl-seconds", "900", str(status)]
            result = self.run_cli(*arguments)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual(json.loads(result.stdout), [
                arguments, str(self.project / "src" / "serve.py"), arguments[0]])


if __name__ == "__main__":
    unittest.main()
