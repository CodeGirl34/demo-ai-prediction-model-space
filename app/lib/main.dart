import 'package:flutter/material.dart';

import 'api.dart';
import 'screens/overview_screen.dart';
import 'screens/replay_screen.dart';
import 'screens/telemetry_screen.dart';
import 'theme.dart';

void main() => runApp(SpacecraftAiApp(api: Api()));

class SpacecraftAiApp extends StatelessWidget {
  const SpacecraftAiApp({super.key, required this.api});
  final Api api;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Spacecraft AI',
        debugShowCheckedModeBanner: false,
        theme: buildTheme(),
        home: HomeShell(api: api),
      );
}

class HomeShell extends StatefulWidget {
  const HomeShell({super.key, required this.api});
  final Api api;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int tab = 0;

  static const _dests = [
    (Icons.dashboard_outlined, Icons.dashboard, 'Overview'),
    (Icons.flight_land_outlined, Icons.flight_land, 'Landing replay'),
    (Icons.satellite_alt_outlined, Icons.satellite_alt, 'Telemetry'),
  ];

  @override
  Widget build(BuildContext context) {
    final pages = [
      OverviewScreen(api: widget.api, onOpenReplay: () => setState(() => tab = 1)),
      ReplayScreen(api: widget.api),
      TelemetryScreen(api: widget.api),
    ];
    final wide = MediaQuery.sizeOf(context).width >= 900;
    final body = IndexedStack(index: tab, children: pages);
    return Scaffold(
      appBar: AppBar(
        title: const Row(children: [
          Icon(Icons.rocket_launch, color: kMars),
          SizedBox(width: 10),
          Text('Spacecraft AI', style: TextStyle(fontWeight: FontWeight.w700)),
          SizedBox(width: 10),
          Flexible(
            child: Text('anomaly detection · mission phases',
                overflow: TextOverflow.ellipsis, style: TextStyle(color: kMuted, fontSize: 14)),
          ),
        ]),
      ),
      body: wide
          ? Row(children: [
              NavigationRail(
                selectedIndex: tab,
                onDestinationSelected: (i) => setState(() => tab = i),
                labelType: NavigationRailLabelType.all,
                backgroundColor: kSurface,
                destinations: [
                  for (final (o, s, l) in _dests)
                    NavigationRailDestination(icon: Icon(o), selectedIcon: Icon(s), label: Text(l)),
                ],
              ),
              Expanded(child: body),
            ])
          : body,
      bottomNavigationBar: wide
          ? null
          : NavigationBar(
              selectedIndex: tab,
              onDestinationSelected: (i) => setState(() => tab = i),
              destinations: [
                for (final (o, s, l) in _dests) NavigationDestination(icon: Icon(o), selectedIcon: Icon(s), label: l),
              ],
            ),
    );
  }
}
