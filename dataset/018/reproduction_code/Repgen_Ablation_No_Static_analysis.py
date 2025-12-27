import unittest
from unittest.mock import patch

from object_detection.models import faster_rcnn_resnet_v1_feature_extractor_tf1_test

class TestFeatureExtractor(unittest.TestCase):
    @patch('object_detection.utils.tf_version.is_tf2', return_value=False)
    def test_setup_with_incorrect_option(self, mock_is_tf2):
        with self.assertRaises(SystemExit) as context:
            faster_rcnn_resnet_v1_feature_extractor_tf1_test.main(['--use-feature=2020-resolver'])
        
        # Verify the expected exit code
        self.assertEqual(context.exception.code, 1)

if __name__ == '__main__':
    unittest.main()