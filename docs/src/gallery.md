---
htmxoFullWidth: true
---

# Transform Gallery

Each card below builds a [`FusedTransform`](api.md#ReparametrizableDistributions.FusedTransform),
draws a fresh batch of unconstrained parameters, runs `fused_logdensity`, and
shows the result alongside the source.

The cards are served live by the [`ReparametrizableDistributionsWeb`
app](https://github.com/nsiccha/ReparametrizableDistributions.jl/tree/dev/web)
in development, and from committed recordings in production. Reload the
page to resample.

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-rd/gallery" hx-swap="innerHTML">
  <em>Loading transform gallery…</em>
</div>
</div>
```
