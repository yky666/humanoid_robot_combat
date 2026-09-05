# Policy artifacts

The active motions use separate MNN 2.9.5 actors produced from their accepted
canonical training exports:

| Motion | Runtime model | Accepted ONNX SHA-256 | MNN SHA-256 |
| --- | --- | --- | --- |
| Front kick | `front_kick_policy.mnn` | `5138f26d954d66741e5d84b85c8b7594e6d0f59dfe1b69a857b2d9f7e2f85069` | `f025f857f074cd5073b6f7abb4eedf677af126c9e934c6706542aa911ca6d8f3` |
| Hook punch | `hook_punch_policy.mnn` | `62f898df5733d9ee31860994cf17ad94c02229f9d4ae2c7164a8aa809925c44f` | `f486411792b0744922e195b18c4f2fff09c7ff9ef119a7789af86a27a9195e4b` |
| Left jab | `jab_left_policy.mnn` | `7b13114adaca9467c95860a82c98d0171ff122cb3aebbfd8a6ac2796e98c092e` | `a0437216bb8d7d9f339840a804c34ee5c874ab6c193ff1072887aa9a51695697` |
| Straight punch | `straight_punch_policy.mnn` | `f0aebc1bed7192be0cb861bad7693063fe87ec014d8daf932aecae3b82e2cf0d` | `7a863b258a5700942628a7ced386f67d9adfd9eb236dbe14ed082f7b91a4b1fa` |

`tools/prepare_qualifier_policy.py` extracts each `obs[1,140] ->
actions[1,25]` actor before conversion. Each MNN passed five deterministic
ONNX-vs-MNN inference comparisons at a `0.001` threshold.

The `t800_qualifier_joint_policy.*` files are retained only to audit the first
hardware smoke test. That actor came from the older six-motion joint run
`2026-09-02_12-44-27_approved_qualifier_6_t800_joint_v1_from31000` and is not
referenced by the corrected active-motion configuration.
