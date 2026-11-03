import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import baseline_vllm as bv  # noqa: E402


def request(new_tokens=3, prompt_len=4, arrival=1.0):
    return {"id": 7, "prompt": list(range(prompt_len)), "max_new_tokens": new_tokens, "arrival": arrival}


def event(completion_tokens):
    return {"choices": [], "usage": {"completion_tokens": completion_tokens}}


class Payload(unittest.TestCase):
    def test_sends_token_ids_greedily_to_the_full_length(self):
        p = bv.completion_payload(request(), "m")
        self.assertEqual(p["prompt"], [0, 1, 2, 3])
        self.assertEqual((p["max_tokens"], p["min_tokens"]), (3, 3))
        self.assertEqual(p["temperature"], 0)
        self.assertTrue(p["ignore_eos"])
        self.assertTrue(p["stream_options"]["continuous_usage_stats"])


class ParseSse(unittest.TestCase):
    def test_reads_data_lines(self):
        self.assertEqual(bv.parse_sse(b'data: {"usage": null}\n'), {"usage": None})

    def test_skips_blank_lines_and_done(self):
        self.assertIsNone(bv.parse_sse(b"\n"))
        self.assertIsNone(bv.parse_sse(b"data: [DONE]\n"))


class TokenCounting(unittest.TestCase):
    def test_times_first_and_last_token_by_usage_not_chunks(self):
        clock = bv.TokenClock()
        clock.observe(event(1), 10.0)
        clock.observe(event(1), 10.5)
        clock.observe(event(3), 11.0)
        clock.observe(event(3), 12.0)
        self.assertEqual((clock.first, clock.last, clock.tokens), (10.0, 11.0, 3))
        self.assertEqual((clock.chunks, clock.multi_token_chunks), (2, 1))

    def test_ignores_chunks_without_usage(self):
        clock = bv.TokenClock()
        clock.observe({"choices": [], "usage": None}, 1.0)
        clock.observe({"choices": []}, 2.0)
        self.assertEqual(clock.tokens, 0)
        self.assertIsNone(clock.first)

    def test_metric_is_relative_to_the_scheduled_arrival(self):
        clock = bv.TokenClock()
        clock.observe(event(1), 102.0)
        clock.observe(event(3), 104.0)
        m = bv.request_metric(request(), clock, 100.0)
        self.assertEqual(m["ttft_s"], 1.0)
        self.assertEqual(m["latency_s"], 3.0)
        self.assertEqual(m["tpot_s"], 1.0)

    def test_short_stream_is_an_error(self):
        clock = bv.TokenClock()
        clock.observe(event(2), 1.0)
        with self.assertRaisesRegex(RuntimeError, "2 tokens streamed, 3 asked for"):
            bv.request_metric(request(), clock, 0.0)


class Stats(unittest.TestCase):
    def test_client_stats(self):
        clock = bv.TokenClock()
        clock.observe(event(2), 1.0)
        s = bv.client_stats([0.0, 0.5], [clock])
        self.assertEqual(s["client_lag_s_max"], 0.5)
        self.assertTrue(math.isclose(s["client_lag_s_p99"], 0.495))
        self.assertEqual((s["stream_chunks"], s["multi_token_chunks"]), (1, 1))

    def test_preemptions_sum_across_labels(self):
        text = ('# TYPE vllm:num_preemptions_total counter\n'
                'vllm:num_preemptions_total{engine="0",model_name="m"} 2.0\n'
                'vllm:num_preemptions_total{engine="1",model_name="m"} 1.0\n')
        self.assertEqual(bv.parse_preemptions(text), 3)
        self.assertEqual(bv.parse_preemptions(""), 0)


class ServerLog(unittest.TestCase):
    def test_reads_dtype_and_backend(self):
        fields = bv.parse_server_log("Initializing with config: dtype=torch.bfloat16, x\n"
                                     "Using FLASH_ATTN attention backend out of potential\n")
        self.assertEqual(fields, {"dtype": "bfloat16", "attention_backend": "FLASH_ATTN"})

    def test_missing_fields_are_empty(self):
        self.assertEqual(bv.parse_server_log("nothing")["dtype"], "")


class Options(unittest.TestCase):
    def test_server_command_fixes_the_dtype_and_batch(self):
        cmd = bv.server_command("vllm", "models/m", 8123, 16, 0.7, 2048)
        self.assertEqual(cmd[:3], ["vllm", "serve", "models/m"])
        self.assertEqual(cmd[cmd.index("--dtype") + 1], "bfloat16")
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "16")

    def test_rates(self):
        self.assertEqual(bv.parse_rates("1,2.5"), [1.0, 2.5])
        with self.assertRaises(ValueError):
            bv.parse_rates("1,0")
        self.assertEqual(bv.rate_name(2.5), "2p5")
        self.assertEqual(bv.rate_name(8.0), "8")


if __name__ == "__main__":
    unittest.main()
