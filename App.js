/**
 * App.js
 * ========
 * Office Attendance Management System -- Employee Mobile App (Expo / React Native)
 *
 * Screens:
 *   1. Login          -- name + password -> POST /token
 *   2. Home            -- personal attendance stats -> GET /attendance/stats/me
 *   3. Apply for Leave -- date pickers -> POST /leaves/, plus leave history -> GET /leaves/me
 *
 * This is a single self-contained file (as requested) using simple
 * state-based screen switching instead of a navigation library -- for an
 * app this size that keeps the dependency list short. If this grows past
 * ~4-5 screens, swap in @react-navigation/native.
 *
 * --------------------------------------------------------------------
 * SETUP
 * --------------------------------------------------------------------
 * 1. Create the project (skip if you already have one):
 *      npx create-expo-app attendance-app
 *      cd attendance-app
 *
 * 2. Install dependencies:
 *      npx expo install expo-secure-store @react-native-community/datetimepicker
 *      npm install axios
 *
 * 3. Replace the generated App.js with this file.
 *
 * 4. Set API_BASE_URL below to your computer's LAN IP (see the comment
 *    right above the constant -- "localhost" will NOT work from a phone).
 *
 * 5. Run it:
 *      npx expo start
 *    Scan the QR code with the Expo Go app on your phone. Your phone and
 *    your computer must be on the SAME Wi-Fi network.
 * --------------------------------------------------------------------
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Alert,
  Platform,
} from 'react-native';
import * as SecureStore from 'expo-secure-store';
import DateTimePicker from '@react-native-community/datetimepicker';
import axios from 'axios';

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------

// Replace YOUR_LOCAL_IP with your computer's LAN IP address -- NOT
// "localhost" or "127.0.0.1". A physical phone can't reach your laptop's
// own loopback address; it needs the laptop's address on the shared Wi-Fi
// network instead.
//
// Find your LAN IP:
//   Windows:      ipconfig            -> look for "IPv4 Address"
//   macOS:        ipconfig getifaddr en0
//   Linux:        hostname -I
//
// Also make sure main.py is actually reachable from other devices, i.e.
// started with --host 0.0.0.0 (or 127.0.0.1 is fine too since uvicorn
// binds all interfaces by default when you pass your machine's real IP
// here -- the important part is which IP the PHONE dials, not what
// uvicorn is told to bind to).
const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';

const SECURE_STORE_TOKEN_KEY = 'attendance_jwt_token';
const SECURE_STORE_ROLE_KEY = 'attendance_role';
const SECURE_STORE_NAME_KEY = 'attendance_name';

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

/**
 * Log in against POST /token.
 *
 * FastAPI's OAuth2PasswordRequestForm expects
 * application/x-www-form-urlencoded, NOT JSON -- so this builds a
 * URLSearchParams body and sets the Content-Type explicitly instead of
 * letting axios default to JSON.
 */
async function apiLogin(username, password) {
  const body = new URLSearchParams();
  body.append('username', username);
  body.append('password', password);

  const response = await axios.post(`${API_BASE_URL}/token`, body.toString(), {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    timeout: 8000,
  });
  return response.data; // { access_token, token_type, role, employee_id, name }
}

/** Build an axios request config with the stored JWT attached as a Bearer token. */
function authConfig(token) {
  return { headers: { Authorization: `Bearer ${token}` }, timeout: 8000 };
}

async function fetchAttendanceStats(token) {
  const response = await axios.get(`${API_BASE_URL}/attendance/stats/me`, authConfig(token));
  return response.data;
}

async function submitLeaveRequest(token, startDate, endDate, reason) {
  const payload = {
    start_date: startDate.toISOString(),
    end_date: endDate.toISOString(),
    reason: reason || null,
  };
  const response = await axios.post(`${API_BASE_URL}/leaves/`, payload, authConfig(token));
  return response.data;
}

async function fetchMyLeaves(token) {
  const response = await axios.get(`${API_BASE_URL}/leaves/me`, authConfig(token));
  return response.data;
}

// ---------------------------------------------------------------------------
// Secure token storage
// ---------------------------------------------------------------------------
// expo-secure-store encrypts values at rest (iOS Keychain / Android
// Keystore) -- meaningfully safer than AsyncStorage for a JWT, which is
// effectively a bearer credential for this person's account.

async function saveSession(token, role, name) {
  await SecureStore.setItemAsync(SECURE_STORE_TOKEN_KEY, token);
  await SecureStore.setItemAsync(SECURE_STORE_ROLE_KEY, role);
  await SecureStore.setItemAsync(SECURE_STORE_NAME_KEY, name);
}

async function loadSession() {
  const token = await SecureStore.getItemAsync(SECURE_STORE_TOKEN_KEY);
  const role = await SecureStore.getItemAsync(SECURE_STORE_ROLE_KEY);
  const name = await SecureStore.getItemAsync(SECURE_STORE_NAME_KEY);
  return token ? { token, role, name } : null;
}

async function clearSession() {
  await SecureStore.deleteItemAsync(SECURE_STORE_TOKEN_KEY);
  await SecureStore.deleteItemAsync(SECURE_STORE_ROLE_KEY);
  await SecureStore.deleteItemAsync(SECURE_STORE_NAME_KEY);
}

// ---------------------------------------------------------------------------
// Screen: Login
// ---------------------------------------------------------------------------

function LoginScreen({ onLoginSuccess }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!username || !password) {
      setError('Please enter both your name and password.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await apiLogin(username, password);
      await saveSession(data.access_token, data.role, data.name);
      onLoginSuccess({ token: data.access_token, role: data.role, name: data.name });
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setError('Incorrect name or password.');
      } else {
        setError(`Could not reach the server at ${API_BASE_URL}. Check your Wi-Fi and API_BASE_URL.`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.centeredContainer}>
        <Text style={styles.title}>🕒 Attendance</Text>
        <Text style={styles.subtitle}>Employee Login</Text>

        <TextInput
          style={styles.input}
          placeholder="Name"
          placeholderTextColor="#64748b"
          autoCapitalize="none"
          value={username}
          onChangeText={setUsername}
        />
        <TextInput
          style={styles.input}
          placeholder="Password"
          placeholderTextColor="#64748b"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />

        {error ? <Text style={styles.errorText}>{error}</Text> : null}

        <TouchableOpacity style={styles.primaryButton} onPress={handleLogin} disabled={loading}>
          {loading ? <ActivityIndicator color="#0f172a" /> : <Text style={styles.primaryButtonText}>Log In</Text>}
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Screen: Home (personal attendance stats)
// ---------------------------------------------------------------------------

function StatRow({ label, value }) {
  return (
    <View style={styles.statRow}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

function HomeScreen({ session, onNavigateToLeave, onLogout }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadStats = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchAttendanceStats(session.token);
      setStats(data);
    } catch (err) {
      if (err.response && err.response.status === 401) {
        Alert.alert('Session expired', 'Please log in again.');
        onLogout();
        return;
      }
      setError('Could not load your attendance stats.');
    } finally {
      setLoading(false);
    }
  }, [session.token, onLogout]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.screenContainer}>
        <Text style={styles.title}>👋 {session.name}</Text>
        <Text style={styles.subtitle}>Your Attendance Summary</Text>

        {loading && <ActivityIndicator size="large" style={{ marginTop: 24 }} />}
        {!!error && <Text style={styles.errorText}>{error}</Text>}

        {stats && !loading && (
          <View style={styles.card}>
            <StatRow label="Total Days Logged" value={stats.total_days_logged} />
            <StatRow label="Present" value={stats.present_count} />
            <StatRow label="Late" value={stats.late_count} />
            <StatRow label="Attendance Rate" value={`${Math.round(stats.attendance_rate * 100)}%`} />
          </View>
        )}

        <TouchableOpacity style={styles.secondaryButton} onPress={loadStats}>
          <Text style={styles.secondaryButtonText}>Refresh</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.primaryButton} onPress={onNavigateToLeave}>
          <Text style={styles.primaryButtonText}>Apply for Leave</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.logoutButton} onPress={onLogout}>
          <Text style={styles.logoutButtonText}>Log Out</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Screen: Apply for Leave
// ---------------------------------------------------------------------------

function ApplyLeaveScreen({ session, onBack }) {
  const [startDate, setStartDate] = useState(new Date());
  const [endDate, setEndDate] = useState(new Date());
  const [showStartPicker, setShowStartPicker] = useState(false);
  const [showEndPicker, setShowEndPicker] = useState(false);
  const [reason, setReason] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const leaves = await fetchMyLeaves(session.token);
      setHistory(leaves);
    } catch (err) {
      // Non-fatal -- the form above still works even if history fails to load.
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, [session.token]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleSubmit = async () => {
    if (endDate < startDate) {
      Alert.alert('Invalid dates', 'End date cannot be before the start date.');
      return;
    }
    setSubmitting(true);
    try {
      await submitLeaveRequest(session.token, startDate, endDate, reason);
      Alert.alert('Success', 'Leave request submitted. Awaiting admin approval.');
      setReason('');
      loadHistory();
    } catch (err) {
      Alert.alert('Error', 'Could not submit your leave request. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.screenContainer}>
        <Text style={styles.title}>📝 Apply for Leave</Text>

        <Text style={styles.label}>Start Date</Text>
        <TouchableOpacity style={styles.dateButton} onPress={() => setShowStartPicker(true)}>
          <Text style={styles.dateButtonText}>{startDate.toDateString()}</Text>
        </TouchableOpacity>
        {showStartPicker && (
          <DateTimePicker
            value={startDate}
            mode="date"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={(event, selectedDate) => {
              setShowStartPicker(Platform.OS === 'ios');
              if (selectedDate) setStartDate(selectedDate);
            }}
          />
        )}

        <Text style={styles.label}>End Date</Text>
        <TouchableOpacity style={styles.dateButton} onPress={() => setShowEndPicker(true)}>
          <Text style={styles.dateButtonText}>{endDate.toDateString()}</Text>
        </TouchableOpacity>
        {showEndPicker && (
          <DateTimePicker
            value={endDate}
            mode="date"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={(event, selectedDate) => {
              setShowEndPicker(Platform.OS === 'ios');
              if (selectedDate) setEndDate(selectedDate);
            }}
          />
        )}

        <Text style={styles.label}>Reason (optional)</Text>
        <TextInput
          style={[styles.input, styles.multilineInput]}
          placeholder="e.g. Family event"
          placeholderTextColor="#64748b"
          multiline
          value={reason}
          onChangeText={setReason}
        />

        <TouchableOpacity style={styles.primaryButton} onPress={handleSubmit} disabled={submitting}>
          {submitting ? <ActivityIndicator color="#0f172a" /> : <Text style={styles.primaryButtonText}>Submit Request</Text>}
        </TouchableOpacity>

        <TouchableOpacity style={styles.secondaryButton} onPress={onBack}>
          <Text style={styles.secondaryButtonText}>Back</Text>
        </TouchableOpacity>

        <Text style={[styles.subtitle, { marginTop: 32 }]}>Your Leave History</Text>
        {historyLoading && <ActivityIndicator style={{ marginTop: 12 }} />}
        {!historyLoading && history.length === 0 && (
          <Text style={styles.mutedText}>No leave requests yet.</Text>
        )}
        {!historyLoading &&
          history.map((leave) => (
            <View key={leave.id} style={styles.leaveRow}>
              <Text style={styles.leaveDates}>
                {new Date(leave.start_date).toLocaleDateString()} -- {new Date(leave.end_date).toLocaleDateString()}
              </Text>
              <Text style={statusBadgeStyle(leave.status)}>{leave.status}</Text>
            </View>
          ))}
      </ScrollView>
    </SafeAreaView>
  );
}

function statusBadgeStyle(status) {
  const color = status === 'Approved' ? '#22c55e' : status === 'Rejected' ? '#ef4444' : '#f59e0b';
  return [styles.leaveStatus, { color }];
}

// ---------------------------------------------------------------------------
// Root App -- simple state-based screen switcher (no navigation library
// needed for 3 screens; swap in @react-navigation/native if this grows).
// ---------------------------------------------------------------------------

export default function App() {
  const [screen, setScreen] = useState('loading'); // 'loading' | 'login' | 'home' | 'leave'
  const [session, setSession] = useState(null); // { token, role, name }

  useEffect(() => {
    (async () => {
      const existing = await loadSession();
      if (existing) {
        setSession(existing);
        setScreen('home');
      } else {
        setScreen('login');
      }
    })();
  }, []);

  const handleLoginSuccess = (newSession) => {
    setSession(newSession);
    setScreen('home');
  };

  const handleLogout = async () => {
    await clearSession();
    setSession(null);
    setScreen('login');
  };

  if (screen === 'loading') {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.centeredContainer}>
          <ActivityIndicator size="large" />
        </View>
      </SafeAreaView>
    );
  }

  if (screen === 'login') {
    return <LoginScreen onLoginSuccess={handleLoginSuccess} />;
  }

  if (screen === 'leave') {
    return <ApplyLeaveScreen session={session} onBack={() => setScreen('home')} />;
  }

  return <HomeScreen session={session} onNavigateToLeave={() => setScreen('leave')} onLogout={handleLogout} />;
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#0f172a' },
  centeredContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
  screenContainer: { flexGrow: 1, padding: 24, paddingTop: 48 },
  title: { fontSize: 28, fontWeight: '700', color: '#f8fafc', marginBottom: 4 },
  subtitle: { fontSize: 16, color: '#94a3b8', marginBottom: 24 },
  label: { fontSize: 14, color: '#cbd5e1', marginTop: 16, marginBottom: 6 },
  mutedText: { color: '#64748b', marginTop: 8 },
  input: {
    width: '100%',
    backgroundColor: '#1e293b',
    color: '#f8fafc',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  multilineInput: { height: 90, textAlignVertical: 'top' },
  dateButton: {
    backgroundColor: '#1e293b',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: '#334155',
    marginBottom: 4,
  },
  dateButtonText: { color: '#f8fafc' },
  primaryButton: {
    backgroundColor: '#38bdf8',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
    marginTop: 20,
    width: '100%',
  },
  primaryButtonText: { color: '#0f172a', fontWeight: '700', fontSize: 16 },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderRadius: 8,
    paddingVertical: 12,
    alignItems: 'center',
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#334155',
    width: '100%',
  },
  secondaryButtonText: { color: '#cbd5e1', fontWeight: '600' },
  logoutButton: { alignItems: 'center', marginTop: 24 },
  logoutButtonText: { color: '#ef4444', fontWeight: '600' },
  errorText: { color: '#ef4444', marginBottom: 12, textAlign: 'center' },
  card: {
    backgroundColor: '#1e293b',
    borderRadius: 12,
    padding: 20,
    marginTop: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#334155',
  },
  statLabel: { color: '#94a3b8', fontSize: 15 },
  statValue: { color: '#f8fafc', fontSize: 15, fontWeight: '700' },
  leaveRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1e293b',
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginTop: 8,
    borderWidth: 1,
    borderColor: '#334155',
  },
  leaveDates: { color: '#f8fafc', fontSize: 13 },
  leaveStatus: { fontWeight: '700', fontSize: 13 },
});
