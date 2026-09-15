import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticContractTests(unittest.TestCase):
    def test_required_audit_docs_exist(self) -> None:
        for rel_path in [
            "docs/PROJECT_AUDIT.md",
            "docs/GAP_ANALYSIS.md",
            "docs/MODEL_AND_GPU_REPORT.md",
            "docs/DEPLOYMENT_STRATEGY.md",
            "docs/REPOSITORY_STRUCTURE.md",
            "docs/IMPROVEMENTS_ROI.md",
            "deployment/README.md",
            "configs/voicelab.example.yaml",
        ]:
            with self.subTest(rel_path=rel_path):
                self.assertTrue((ROOT / rel_path).is_file())

    def test_detection_checkpoints_are_packaged(self) -> None:
        for name in ["ModelA_LA_bestnew.pt", "ModelB_PA_bestnew.pt"]:
            path = ROOT / "models" / "checkpoints" / name
            with self.subTest(name=name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 100_000)

    def test_webapp_defaults_to_detection_checkpoint_dir(self) -> None:
        main_py = (ROOT / "webapp" / "main.py").read_text(encoding="utf-8")
        self.assertIn('models" / "checkpoints', main_py)
        self.assertIn("VOICELAB_ENABLE_TTS", main_py)
        self.assertIn("TTS_IMPORT_ERROR", main_py)
        self.assertIn("generation_enabled", main_py)

    def test_tts_dependencies_are_optional(self) -> None:
        runtime_requirements = (ROOT / "webapp" / "requirements.txt").read_text(encoding="utf-8")
        tts_requirements = (ROOT / "webapp" / "requirements-tts.txt").read_text(encoding="utf-8")
        self.assertNotIn("coqui-tts", runtime_requirements)
        self.assertIn("coqui-tts", tts_requirements)


if __name__ == "__main__":
    unittest.main()

