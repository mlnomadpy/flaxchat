# Read-only TPU diagnosis

Checkpoint3000 restored on8 physical TPU devices. All parameters finite, maxabsolute1.668391. Exact first8 heldout examples:globalbatch8 lossNaN; duplicate sameexamples to16 loss2.487855; duplicate64 loss2.487744. Embeddingnorm finite,max11.086; separatelyjitted wholelayer0 first nonfinite atbatch8. This localizes a shape-dependent forward failure; it does not identify attention versusFFN or prove a compilerbug. No arithmetic change or nanreplacement was made. Onlycheckpoint3000 produced a completed diagnostic receipt; older2500/2750 probes were not completed.

Diagnostic completed with scope cleanupverified and independent zone lists empty. No checkpoint writes/optimizer updates. Later recoveryD qualifies full512heldout rows atbatch16 before trainingmicrobatches16; batch8 remains unvalidated/failing and blocks portable release acceptance atthatshape.
