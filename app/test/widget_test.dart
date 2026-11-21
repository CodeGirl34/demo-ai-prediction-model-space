import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:spacecraft_ai_app/api.dart';
import 'package:spacecraft_ai_app/main.dart';
import 'package:spacecraft_ai_app/theme.dart';

void main() {
  test('time and altitude formatting', () {
    expect(fmtT(-12.34), 'T−12.3 s');
    expect(fmtT(5), 'T+5.0 s');
    expect(fmtAlt(125.4), '125 km');
    expect(fmtAlt(0.25), '250 m');
  });

  testWidgets('shows an error with retry when the API is down', (tester) async {
    final api = Api(base: 'http://test', client: MockClient((_) async => http.Response('down', 503)));
    await tester.pumpWidget(SpacecraftAiApp(api: api));
    await tester.pumpAndSettle();
    expect(find.text('SPACECRAFT AI'), findsOneWidget);
    expect(find.text('Could not reach the API'), findsWidgets);
    expect(find.widgetWithText(FilledButton, 'Retry'), findsWidgets);
  });
}
