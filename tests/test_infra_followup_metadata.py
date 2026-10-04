"""Infrastructure workflow and provider identity checks, without model execution."""

import unittest
from unittest.mock import patch
from scripts import representation_workflow as workflow
from scripts.gcp_spot_supervisor import billing_label, verify_node_labels
from scripts.representation_workflow import validate_workflow


class InfraFollowupTests(unittest.TestCase):
    def test_provider_labels_keep_valid_names_and_hash_normalized_names(self):
        self.assertEqual(billing_label("yat-stage-1"), "yat-stage-1")
        self.assertNotEqual(billing_label("A B"), billing_label("a-b"))
        self.assertLessEqual(len(billing_label("a" * 64)), 63)
        self.assertRegex(billing_label("A B"), r"^[a-z0-9_-]+$")
        labels = {"flaxchat-run": "run", "flaxchat-stage": "stage"}
        self.assertEqual(verify_node_labels({"labels": labels}, labels), labels)
        with self.assertRaises(ValueError):
            verify_node_labels({"labels": {"flaxchat-run": "other"}}, labels)
        with self.assertRaises(ValueError):
            verify_node_labels(None, labels)

    def test_miracl_binds_shard_and_hash_artifact(self):
        value = dict(
            workflow_contract=dict(
                kind="miracl", shard_index=3, subsets_target="shard.json"
            ),
            workload=[
                "{python}",
                "-m",
                "scripts.evaluate_yat_mteb_subsets",
                "--task",
                "MIRACLRetrievalHardNegatives",
                "--subsets-file",
                "{root}/shard.json",
            ],
            artifacts=[{"target": "shard.json"}],
        )
        validate_workflow(value, "miracl", 3)
        with self.assertRaises(ValueError):
            validate_workflow(value, "miracl", 4)
        with self.assertRaises(ValueError):
            validate_workflow({**value, "artifacts": []}, "miracl", 3)

    def test_publication_binds_repo_release_and_module(self):
        for kind, module in [
            ("publish-flax", "scripts.publish_yat_embedding_from_gcp"),
            ("publish-torch", "scripts.publish_yat_torch_from_gcp"),
        ]:
            value = dict(
                workflow_contract=dict(
                    kind=kind, repo="owner/new-model", release_target="release"
                ),
                workload=[
                    "{python}",
                    "-m",
                    module,
                    "{root}/release",
                    "/tmp/token",
                    "--repo",
                    "owner/new-model",
                ],
                artifacts=[{"target": "release"}],
            )
            if kind == "publish-torch":
                value["workflow_contract"]["parity_evidence_target"] = "private-parity"
                value["workload"][3:3] = ["--parity-evidence", "{root}/private-parity"]
                value["workload"][3:3] = [
                    "--host-ram-budget-bytes",
                    str(16 * 2**30),
                    "--evidence-budget-bytes",
                    str(16 * 2**30),
                ]
                value["artifacts"].append({"target": "private-parity"})
            validate_workflow(value, kind)
            with self.assertRaises(ValueError):
                validate_workflow(
                    {**value, "workload": value["workload"][:-1] + ["owner/old-model"]},
                    kind,
                )
            with self.assertRaises(ValueError):
                validate_workflow({**value, "artifacts": []}, kind)
            if kind == "publish-torch":
                old = {
                    **value,
                    "workload": [
                        arg
                        for arg in value["workload"]
                        if arg not in ("--parity-evidence", "{root}/private-parity")
                    ],
                }
                with self.assertRaises(ValueError):
                    validate_workflow(old, kind)

    def test_wrapper_setup_then_run_and_failure_stops_execution(self):
        value = dict(
            workflow_contract=dict(
                kind="publish-flax", repo="owner/model", release_target="release"
            ),
            workload=[
                "{python}",
                "-m",
                "scripts.publish_yat_embedding_from_gcp",
                "{root}/release",
                "/tmp/token",
                "--repo",
                "owner/model",
            ],
            artifacts=[{"target": "release"}],
        )
        args = [
            "--kind",
            "publish-flax",
            "--manifest",
            "/tmp/run.json",
            "--root",
            "/tmp/root",
        ]
        with (
            patch.object(workflow.runner, "load_manifest", return_value=value),
            patch.object(workflow.runner, "setup") as setup,
            patch.object(workflow.runner, "run_worker", return_value=7) as run,
        ):
            self.assertEqual(workflow.main(args), 7)
            setup.assert_called_once()
            run.assert_called_once()
        with (
            patch.object(workflow.runner, "load_manifest", return_value=value),
            patch.object(
                workflow.runner, "setup", side_effect=ValueError("hash failure")
            ),
            patch.object(workflow.runner, "run_worker") as run,
        ):
            with self.assertRaises(ValueError):
                workflow.main(args)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
