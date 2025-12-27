import unittest
from object_detection.utils import test_case

class TestFeatureFlag(test_case.TestCase):
    @unittest.skipIf(not hasattr(tf, '__version__') and tf.__version__.startswith('2'), "Test requires TensorFlow 2.x")
    def test_use_feature_flag(self):
        with self.assertRaisesRegex(
            Exception,
            r"invalid choice: '2020-resolver'"
        ):
            # Note: The --use-feature option is not valid for pip version < 20.3
            # This will fail if pip version is less than 20.3, which might be the case in some environments
            self.run_command("python -m pip install --use-feature=2020-resolver .")

if __name__ == '__main__':
    unittest.main()