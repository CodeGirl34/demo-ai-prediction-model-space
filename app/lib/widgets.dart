import 'package:flutter/material.dart';

import 'theme.dart';

class SectionCard extends StatelessWidget {
  const SectionCard({super.key, required this.child, this.title, this.trailing, this.padding = 16});
  final Widget child;
  final String? title;
  final Widget? trailing;
  final double padding;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: EdgeInsets.all(padding),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            if (title != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 12),
                child: Row(children: [
                  Expanded(
                    child: Text(title!,
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600)),
                  ),
                  ?trailing,
                ]),
              ),
            child,
          ],
        ),
      ),
    );
  }
}

class Metric extends StatelessWidget {
  const Metric({super.key, required this.label, required this.value, this.color, this.caption});
  final String label;
  final String value;
  final Color? color;
  final String? caption;

  @override
  Widget build(BuildContext context) {
    final t = Theme.of(context).textTheme;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
      Text(label, style: t.labelMedium?.copyWith(color: kMuted)),
      const SizedBox(height: 2),
      Text(value, style: t.headlineSmall?.copyWith(fontWeight: FontWeight.w700, color: color)),
      if (caption != null) Text(caption!, style: t.bodySmall?.copyWith(color: kMuted)),
    ]);
  }
}

class PhaseChip extends StatelessWidget {
  const PhaseChip(this.phase, {super.key, this.large = false});
  final String phase;
  final bool large;

  @override
  Widget build(BuildContext context) {
    final c = phaseColors[phase] ?? kMuted;
    return Container(
      padding: EdgeInsets.symmetric(horizontal: large ? 14 : 8, vertical: large ? 6 : 3),
      decoration: BoxDecoration(
        color: c.withValues(alpha: 0.18),
        border: Border.all(color: c),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(phaseLabel(phase),
          style: TextStyle(color: c, fontWeight: FontWeight.w600, fontSize: large ? 16 : 12)),
    );
  }
}

class Tag extends StatelessWidget {
  const Tag(this.text, {super.key, required this.color});
  final String text;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
        decoration: BoxDecoration(color: color.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(6)),
        child: Text(text, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600)),
      );
}

class LoadingOrError extends StatelessWidget {
  const LoadingOrError({super.key, this.error, this.onRetry});
  final Object? error;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    if (error == null) return const Center(child: Padding(padding: EdgeInsets.all(32), child: CircularProgressIndicator()));
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.cloud_off, size: 40, color: kMuted),
          const SizedBox(height: 8),
          Text('Could not reach the API', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          Text('$error', textAlign: TextAlign.center, style: const TextStyle(color: kMuted)),
          if (onRetry != null) ...[
            const SizedBox(height: 12),
            FilledButton.tonal(onPressed: onRetry, child: const Text('Retry')),
          ],
        ]),
      ),
    );
  }
}

/// Coloured strip showing phases along the time axis.
class PhaseStrip extends StatelessWidget {
  const PhaseStrip({super.key, required this.phases, this.height = 14, this.upTo});
  final List<String> phases;
  final double height;
  final int? upTo;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: height,
      child: CustomPaint(painter: _StripPainter(phases, upTo), size: Size.infinite),
    );
  }
}

class _StripPainter extends CustomPainter {
  _StripPainter(this.phases, this.upTo);
  final List<String> phases;
  final int? upTo;

  @override
  void paint(Canvas canvas, Size size) {
    if (phases.isEmpty) return;
    final n = phases.length;
    final w = size.width / n;
    int start = 0;
    for (int i = 1; i <= n; i++) {
      if (i == n || phases[i] != phases[start]) {
        final visible = upTo == null || start <= upTo!;
        final c = (phaseColors[phases[start]] ?? kMuted).withValues(alpha: visible ? 0.9 : 0.15);
        canvas.drawRect(Rect.fromLTWH(start * w, 0, (i - start) * w + 0.5, size.height), Paint()..color = c);
        start = i;
      }
    }
  }

  @override
  bool shouldRepaint(covariant _StripPainter old) => old.phases != phases || old.upTo != upTo;
}
