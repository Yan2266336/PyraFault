"""The shared Hugging Face wrapper the VLM baselines sit on."""
from __future__ import annotations

from pyrafault.verify.vlm_baseline import _HfVlm

class TestAReasoningTemplateIsSwitchedOff:
    """Qwen3.8-27B cost an entire ladder row to this.

    Its chat template opens every assistant turn with `<think>\\n`, so with the default
    64 tokens the reply was 100% reasoning prose and 0% answer: 100 unparsed replies on
    each of the seven presentations, and a row reading AUROC 0.49-0.50 -- exactly what a
    constant score yields, and indistinguishable from a model that cannot do the task.
    The captured reply shows it describing the tissue accurately ("a papillary or
    glandular structure ... vacuolated cytoplasm") and never reaching a coordinate.

    The same trap had already been found and fixed for the agent loop without the fix
    reaching the verifier baselines, which is why these tests pin it on the shared base.
    """

    def test_a_known_reasoning_model_is_detected_from_its_id(self):
        v = _HfVlm("Qwen/Qwen3.8-27B", _defer_load=True)
        assert v._template_opens_thinking() is True

    def test_an_unknown_model_whose_template_opens_think_is_detected(self):
        """The id list can only name what was known when it was written, so the rendered
        template is consulted too."""
        v = _HfVlm("some-lab/brand-new-reasoner", _defer_load=True)
        v._proc = type("P", (), {"apply_chat_template": staticmethod(
            lambda *a, **k: "<|im_start|>assistant\n<think>\n")})()
        assert v._template_opens_thinking() is True

    def test_an_ordinary_model_is_not(self):
        v = _HfVlm("google/medgemma-1.5-4b-it", _defer_load=True)
        v._proc = type("P", (), {"apply_chat_template": staticmethod(
            lambda *a, **k: "<start_of_turn>model\n")})()
        assert v._template_opens_thinking() is False

    def test_an_explicit_flag_is_not_overridden(self):
        v = _HfVlm("Qwen/Qwen3.8-27B", _defer_load=True, no_thinking=False)
        assert v._no_thinking is False

    def test_a_template_that_refuses_the_keyword_still_runs(self):
        """A model with no reasoning mode must be unaffected rather than fail to render.
        The fallback is why a mis-detection costs nothing."""
        calls = []

        class Proc:
            def apply_chat_template(self, msg, **kw):
                calls.append(kw)
                if "enable_thinking" in kw:
                    raise TypeError("unexpected keyword 'enable_thinking'")
                # A real processor returns a BatchEncoding, which the caller moves to
                # the device; a bare dict has no `.to`, so the stand-in needs one.
                import torch

                class Batch(dict):
                    def to(self, _device):
                        return self

                return Batch(input_ids=torch.zeros((1, 3), dtype=torch.long))

            def decode(self, *a, **k):
                return "ANSWER"

        class Model:
            device = "cpu"

            def generate(self, **kw):
                import torch
                return torch.zeros((1, 4), dtype=int)

        v = _HfVlm("Qwen/Qwen3.8-27B", _defer_load=True, no_thinking=True)
        v._proc, v._model = Proc(), Model()
        assert v._ask([], "p") == "ANSWER"
        assert "enable_thinking" in calls[0] and "enable_thinking" not in calls[1]
