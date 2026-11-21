import 'package:flutter/material.dart';

const kBackground = Color(0xFF0B0F1A);
const kSurface = Color(0xFF141A2A);
const kSurfaceHigh = Color(0xFF1C2438);
const kMars = Color(0xFFFF6B35);
const kOk = Color(0xFF4CAF50);
const kAlarm = Color(0xFFFFB300);
const kFault = Color(0xFFE53935);
const kMuted = Color(0xFF8A93A8);

const phaseColors = <String, Color>{
  'approach': Color(0xFF9E9E9E),
  'entry': Color(0xFFE4572E),
  'parachute': Color(0xFF29A3D6),
  'powered_descent': Color(0xFFF3A712),
  'sky_crane': Color(0xFF4CAF50),
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

const channelLabels = <String, String>{
  'vertical_speed_m_s': 'Vertical speed (m/s)',
  'sensed_accel_m_s2': 'IMU acceleration (m/s²)',
};

ThemeData buildTheme() {
  final base = ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: ColorScheme.fromSeed(
      seedColor: kMars,
      brightness: Brightness.dark,
      surface: kSurface,
    ),
    scaffoldBackgroundColor: kBackground,
  );
  return base.copyWith(
    cardTheme: const CardThemeData(color: kSurface, elevation: 0, margin: EdgeInsets.zero),
    appBarTheme: const AppBarTheme(backgroundColor: kBackground, elevation: 0, centerTitle: false),
    navigationBarTheme: const NavigationBarThemeData(backgroundColor: kSurface),
    chipTheme: base.chipTheme.copyWith(side: BorderSide.none),
  );
}

String phaseLabel(String p) => phaseLabels[p] ?? p;

String fmtT(num t) {
  final s = t.abs();
  return t <= 0 ? 'T−${s.toStringAsFixed(1)} s' : 'T+${s.toStringAsFixed(1)} s';
}

String fmtAlt(num km) => km >= 1 ? '${km.toStringAsFixed(km >= 100 ? 0 : 1)} km' : '${(km * 1000).toStringAsFixed(0)} m';
