import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

/// Base URL of the FastAPI backend.
///
/// * Web build served by the API itself: same origin (default).
/// * Mobile / desktop: pass `--dart-define=API_BASE=http://<host>:8000`
///   (iOS simulator and macOS: http://localhost:8000, Android emulator: http://10.0.2.2:8000).
String apiBase() {
  const fromEnv = String.fromEnvironment('API_BASE');
  if (fromEnv.isNotEmpty) return fromEnv;
  if (kIsWeb) return Uri.base.origin;
  return 'http://localhost:8000';
}

class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

class Api {
  Api({String? base, http.Client? client})
      : base = base ?? apiBase(),
        _client = client ?? http.Client();

  final String base;
  final http.Client _client;

  Future<dynamic> _get(String path) async {
    final r = await _client.get(Uri.parse('$base$path'));
    return _decode(r);
  }

  Future<dynamic> _post(String path, Map<String, dynamic> body) async {
    final r = await _client.post(Uri.parse('$base$path'),
        headers: {'Content-Type': 'application/json'}, body: jsonEncode(body));
    return _decode(r);
  }

  dynamic _decode(http.Response r) {
    if (r.statusCode != 200) {
      String detail = r.body;
      try {
        detail = jsonDecode(r.body)['detail'].toString();
      } catch (_) {}
      throw ApiException('API ${r.statusCode}: $detail');
    }
    return jsonDecode(utf8.decode(r.bodyBytes));
  }

  Future<Map<String, dynamic>> health() async => await _get('/api/health');
  Future<Map<String, dynamic>> summary() async => await _get('/api/summary');
  Future<List<dynamic>> missions() async => await _get('/api/missions');
  Future<List<dynamic>> telemetryChannels() async => await _get('/api/telemetry/channels');
  Future<Map<String, dynamic>> telemetryChannel(String id) async =>
      await _get('/api/telemetry/channels/$id');
  Future<Map<String, dynamic>> journey() async => await _get('/api/journey');

  Future<Map<String, dynamic>> replay(
    String mission, {
    required String channel,
    List<Map<String, dynamic>> faults = const [],
    int randomFaults = 0,
    int seed = 0,
  }) async =>
      await _post('/api/missions/$mission/replay', {
        'channel': channel,
        'faults': faults,
        'random_faults': randomFaults,
        'seed': seed,
      });
}
