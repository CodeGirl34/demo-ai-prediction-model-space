import 'package:flutter/material.dart';

/// Console design tokens, ported from the RedSight mission console (app/src/styles/tokens.css).
/// World: inside a spacecraft. Deep blue-charcoal hull, thin lit seams, squared instrument
/// panels with a chamfered corner, one amber signal colour, and evidence colours that say
/// where every number comes from.

// -- Surfaces: hull to raised panel ------------------------------------------
const kVoid = Color(0xFF06080B);
const kHull = Color(0xFF0B0F15);
const kPanel = Color(0xFF10151D);
const kPanel2 = Color(0xFF151B25);
const kInset = Color(0xFF0A0E14);
const kSeam = Color(0xFF1E2733);
const kSeamLit = Color(0xFF2C3947);

// -- Ink ------------------------------------------------------------------------
const kInk = Color(0xFFE9EDF3);
const kInkDim = Color(0xFF9AA6B5);
const kInkFaint = Color(0xFF7D8895);

// -- Signal + status ----------------------------------------------------------
const kSignal = Color(0xFFFFB224); // accent: selection, focus, primary action, warnings
const kSignalSoft = Color(0xFF3A2C10);
const kCaution = Color(0xFFFF6D6D); // danger: active anomaly alarm, missed anomaly
const kCautionSoft = Color(0xFF3A1717);
const kGo = Color(0xFF4FD08A); // nominal, caught, AI agrees
const kGoSoft = Color(0xFF11301F);

// -- Evidence: where a number comes from ----------------------------------------
const kObserved = Color(0xFF56D8F4); // NASA data (measured / reconstructed by NASA)
const kObservedSoft = Color(0xFF0E2A33);
const kDerived = Color(0xFFB9A4F5); // AI output (our models)
const kDerivedSoft = Color(0xFF221C3A);
const kScenario = Color(0xFFF59A5F); // simulated (faults we injected)
const kScenarioSoft = Color(0xFF33200F);

// -- Fonts ------------------------------------------------------------------------
const kFontDisplay = 'ChakraPetch';
const kFontBody = 'IBMPlexSans';
const kFontData = 'IBMPlexMono';

const phaseColors = <String, Color>{
  'approach': Color(0xFF8A94A3),
  'entry': Color(0xFFE5484D),
  'parachute': Color(0xFF56D8F4),
  'powered_descent': Color(0xFFFFB224),
  'sky_crane': Color(0xFF4FD08A),
};

const phaseLabels = <String, String>{
  'approach': 'Approach',
  'entry': 'Entry',
  'parachute': 'Parachute',
  'powered_descent': 'Powered descent',
  'sky_crane': 'Sky crane',
};

const faultLabels = <String, String>{
  'spike': 'Spike',
  'bias': 'Bias',
  'drift': 'Drift',
  'stuck': 'Stuck sensor',
  'noise': 'Noise burst',
  'dropout': 'Dropout',
};

/// What each fault does to the reading, in plain words (shown when adding one).
const faultHelp = <String, String>{
  'spike': 'A split-second glitch: one wild reading, then back to normal.',
  'bias': 'The reading jumps by a fixed amount and stays off for a few seconds.',
  'drift': 'The reading slowly creeps away from the true value.',
  'stuck': 'The sensor freezes on its last value. Hard to see when the real value is not changing much.',
  'noise': 'The reading turns jittery for a few seconds, like heavy vibration.',
  'dropout': 'The sensor reads zero. Invisible when the real value is already close to zero.',
};

String channelUnit(String channel) => channel == 'sensed_accel_m_s2' ? 'm/s²' : 'm/s';

const channelLabels = <String, String>{
  'vertical_speed_m_s': 'Vertical speed (m/s)',
  'sensed_accel_m_s2': 'IMU acceleration (m/s²)',
};

/// Uppercase console micro-label (RedSight `.label-console`). Pass the text already upper-cased.
TextStyle consoleLabel({Color color = kInkFaint, double size = 11, FontWeight weight = FontWeight.w500}) =>
    TextStyle(
      fontFamily: kFontDisplay,
      fontWeight: weight,
      fontSize: size,
      letterSpacing: size * 0.14,
      color: color,
      height: 1.2,
    );

/// Tabular numeric readout (RedSight `.data-readout`).
TextStyle readout({Color color = kInk, double size = 13, FontWeight weight = FontWeight.w400}) => TextStyle(
      fontFamily: kFontData,
      fontWeight: weight,
      fontSize: size,
      color: color,
      fontFeatures: const [FontFeature.tabularFigures()],
      height: 1.25,
    );

/// Squared console corner with a 45° nick, top-left + bottom-right (RedSight `.chamfer`).
OutlinedBorder chamfer({double cut = 10, Color side = kSeam, double width = 1}) => BeveledRectangleBorder(
      borderRadius: BorderRadius.only(topLeft: Radius.circular(cut), bottomRight: Radius.circular(cut)),
      side: width == 0 ? BorderSide.none : BorderSide(color: side, width: width),
    );

const kPanelShadow = [BoxShadow(color: Color(0xCC000000), blurRadius: 24, spreadRadius: -12, offset: Offset(0, 8))];

ThemeData buildTheme() {
  const scheme = ColorScheme.dark(
    primary: kSignal,
    onPrimary: Color(0xFF14100A),
    primaryContainer: kSignalSoft,
    onPrimaryContainer: kSignal,
    secondary: kObserved,
    onSecondary: kVoid,
    secondaryContainer: kPanel2,
    onSecondaryContainer: kInk,
    surface: kPanel,
    onSurface: kInk,
    onSurfaceVariant: kInkDim,
    surfaceContainerHighest: kPanel2,
    surfaceContainerHigh: kPanel2,
    surfaceContainer: kPanel,
    surfaceContainerLow: kHull,
    error: kCaution,
    onError: kVoid,
    outline: kSeamLit,
    outlineVariant: kSeam,
  );
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: scheme,
    scaffoldBackgroundColor: kVoid,
    fontFamily: kFontBody,
    splashFactory: NoSplash.splashFactory,
  );
  const square = RoundedRectangleBorder(borderRadius: BorderRadius.zero);
  final buttonLabel = consoleLabel(size: 12, weight: FontWeight.w600);
  return base.copyWith(
    textTheme: base.textTheme.apply(fontFamily: kFontBody, bodyColor: kInk, displayColor: kInk),
    dividerTheme: const DividerThemeData(color: kSeam, thickness: 1, space: 1),
    cardTheme: CardThemeData(color: kPanel, elevation: 0, margin: EdgeInsets.zero, shape: chamfer()),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: kSignal,
        foregroundColor: const Color(0xFF14100A),
        disabledBackgroundColor: kSignalSoft,
        disabledForegroundColor: kInkFaint,
        shape: chamfer(cut: 6, width: 0),
        textStyle: buttonLabel,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        shape: square,
        textStyle: buttonLabel,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      ).copyWith(
        side: WidgetStateProperty.resolveWith((s) => BorderSide(
            color: s.contains(WidgetState.disabled)
                ? kSeam
                : s.contains(WidgetState.hovered)
                    ? kSignal.withValues(alpha: 0.6)
                    : kSeamLit)),
        foregroundColor: WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.disabled)
            ? kInkFaint.withValues(alpha: 0.5)
            : s.contains(WidgetState.hovered)
                ? kSignal
                : kInkDim),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: kSignal, shape: square, textStyle: buttonLabel),
    ),
    segmentedButtonTheme: SegmentedButtonThemeData(
      style: ButtonStyle(
        shape: const WidgetStatePropertyAll(square),
        side: const WidgetStatePropertyAll(BorderSide(color: kSeamLit)),
        textStyle: WidgetStatePropertyAll(consoleLabel(size: 11.5, weight: FontWeight.w600)),
        backgroundColor: WidgetStateProperty.resolveWith(
            (s) => s.contains(WidgetState.selected) ? kSignalSoft : Colors.transparent),
        foregroundColor:
            WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? kSignal : kInkDim),
        iconColor:
            WidgetStateProperty.resolveWith((s) => s.contains(WidgetState.selected) ? kSignal : kInkFaint),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: kInset,
      labelStyle: consoleLabel(size: 11),
      floatingLabelStyle: consoleLabel(size: 11, color: kSignal),
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      border: const OutlineInputBorder(borderRadius: BorderRadius.zero, borderSide: BorderSide(color: kSeamLit)),
      enabledBorder:
          const OutlineInputBorder(borderRadius: BorderRadius.zero, borderSide: BorderSide(color: kSeamLit)),
      focusedBorder:
          const OutlineInputBorder(borderRadius: BorderRadius.zero, borderSide: BorderSide(color: kSignal)),
    ),
    chipTheme: base.chipTheme.copyWith(
      backgroundColor: kScenarioSoft.withValues(alpha: 0.6),
      side: BorderSide(color: kScenario.withValues(alpha: 0.45)),
      shape: square,
      labelStyle: readout(size: 12, color: kInk),
      deleteIconColor: kInkDim,
    ),
    sliderTheme: const SliderThemeData(
      activeTrackColor: kSignal,
      inactiveTrackColor: kSeamLit,
      thumbColor: kSignal,
      overlayColor: Color(0x22FFB224),
      trackHeight: 2,
      thumbShape: RoundSliderThumbShape(enabledThumbRadius: 7),
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: kPanel,
      shape: chamfer(side: kSeamLit),
      titleTextStyle: consoleLabel(size: 14, color: kInk, weight: FontWeight.w600),
    ),
    tooltipTheme: TooltipThemeData(
      decoration: const BoxDecoration(color: kPanel2, border: Border.fromBorderSide(BorderSide(color: kSeamLit))),
      textStyle: readout(size: 11, color: kInkDim),
    ),
    listTileTheme: ListTileThemeData(
      iconColor: kInkFaint,
      titleTextStyle: const TextStyle(fontFamily: kFontBody, color: kInk, fontSize: 13.5, fontWeight: FontWeight.w500),
      subtitleTextStyle: readout(size: 11, color: kInkFaint),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(color: kSignal, linearTrackColor: kSeam),
    scrollbarTheme: ScrollbarThemeData(
      thumbColor: WidgetStatePropertyAll(kSeamLit.withValues(alpha: 0.9)),
      thickness: const WidgetStatePropertyAll(6),
      radius: Radius.zero,
    ),
    popupMenuTheme: PopupMenuThemeData(color: kPanel2, shape: chamfer(cut: 6, side: kSeamLit)),
    textSelectionTheme: const TextSelectionThemeData(cursorColor: kSignal, selectionColor: Color(0x66FFB224)),
  );
}

String phaseLabel(String p) => phaseLabels[p] ?? p;

String fmtT(num t) {
  final s = t.abs();
  return t <= 0 ? 'T−${s.toStringAsFixed(1)} s' : 'T+${s.toStringAsFixed(1)} s';
}

String fmtAlt(num km) => km >= 1 ? '${km.toStringAsFixed(km >= 100 ? 0 : 1)} km' : '${(km * 1000).toStringAsFixed(0)} m';
