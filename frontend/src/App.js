import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import {
  LayoutDashboard,
  GitBranch,
  FlaskConical,
  ShieldCheck,
  ScrollText,
  Settings,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  Activity,
  Database,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ChevronRight,
  Loader2,
  Box,
  Trash2,
  Eye,
  AlertTriangle,
  Zap,
  Wifi,
  WifiOff,
  Brain,
  Target,
  BarChart3,
  Layers,
  Plus,
  Sparkles
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';
const WS_URL = API_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/ws';

// ============================================================================
// WebSocket Hook
// ============================================================================

const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    
    try {
      wsRef.current = new WebSocket(WS_URL);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(connect, 5000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setIsConnected(false);
    }
  }, []);
  
  const sendMessage = useCallback((message) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);
  
  useEffect(() => {
    connect();
    
    // Ping every 30 seconds to keep connection alive
    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' });
    }, 30000);
    
    return () => {
      clearInterval(pingInterval);
      clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect, sendMessage]);
  
  return { isConnected, lastMessage, sendMessage };
};

// ============================================================================
// API Functions
// ============================================================================

const api = {
  async get(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`);
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  },
  async post(endpoint, data = {}) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  },
  async delete(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  }
};

// ============================================================================
// Sidebar Component
// ============================================================================

const Sidebar = ({ activePage, setActivePage, isConnected }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'pipelines', label: 'Pipelines', icon: GitBranch },
    { id: 'experiments', label: 'Experiments', icon: FlaskConical },
    { id: 'automl', label: 'AutoML', icon: Brain },
    { id: 'validations', label: 'Data Quality', icon: ShieldCheck },
    { id: 'logs', label: 'Logs', icon: ScrollText },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <Layers size={20} color="#fafafa" />
        </div>
        <span className="sidebar-title">ETL & ML</span>
        <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`} title={isConnected ? 'Real-time connected' : 'Disconnected'}>
          {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
        </div>
      </div>
      
      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            data-testid={`nav-${item.id}`}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}
          >
            <item.icon size={18} />
            <span>{item.label}</span>
            {item.id === 'automl' && <Sparkles size={12} className="nav-badge" />}
          </button>
        ))}
      </nav>
      
      <div className="sidebar-footer">
        <div className="version-badge">v2.0.0</div>
      </div>
    </aside>
  );
};

// ============================================================================
// Status Badge Component
// ============================================================================

const StatusBadge = ({ status }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'success':
      case 'completed': return <CheckCircle2 size={12} />;
      case 'failed': return <XCircle size={12} />;
      case 'running': return <Loader2 size={12} className="animate-spin" />;
      case 'pending': return <Clock size={12} />;
      default: return <Activity size={12} />;
    }
  };

  const normalizedStatus = status === 'completed' ? 'success' : status;

  return (
    <span className={`status-badge ${normalizedStatus}`} data-testid={`status-badge-${status}`}>
      {getStatusIcon()}
      {status}
    </span>
  );
};

// ============================================================================
// Dashboard Page
// ============================================================================

const DashboardPage = ({ lastMessage }) => {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [statsData, metricsData, runsData] = await Promise.all([
        api.get('/api/dashboard/stats'),
        api.get('/api/dashboard/metrics'),
        api.get('/api/dashboard/recent-runs?limit=5')
      ]);
      setStats(statsData);
      setMetrics(metricsData);
      setRecentRuns(runsData);
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'pipeline_completed' || lastMessage.type === 'experiment_completed') {
        loadData();
      }
    }
  }, [lastMessage, loadData]);

  const COLORS = ['#10b981', '#ef4444'];

  if (loading) {
    return (
      <div className="empty-state">
        <Loader2 size={48} className="animate-spin" />
        <p className="mt-4">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">Dashboard</h1>
            <p className="page-description">Real-time overview of ETL pipelines and ML experiments</p>
          </div>
          <button className="btn btn-secondary" data-testid="refresh-dashboard" onClick={loadData}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Stats Cards */}
        <div className="stat-card" data-testid="stat-total-pipelines">
          <div className="stat-icon-wrapper">
            <GitBranch size={20} />
          </div>
          <div className="stat-label">Total Pipelines</div>
          <div className="stat-value">{stats?.total_pipelines || 0}</div>
          <div className="stat-change positive">
            <TrendingUp size={12} /> Active: {stats?.active_pipelines || 0}
          </div>
        </div>

        <div className="stat-card" data-testid="stat-experiments">
          <div className="stat-icon-wrapper experiments">
            <FlaskConical size={20} />
          </div>
          <div className="stat-label">Experiments</div>
          <div className="stat-value">{stats?.total_experiments || 0}</div>
          <div className="stat-change positive">
            <Box size={12} /> Models: {stats?.total_models || 0}
          </div>
        </div>

        <div className="stat-card" data-testid="stat-success-runs">
          <div className="stat-icon-wrapper success">
            <CheckCircle2 size={20} />
          </div>
          <div className="stat-label">Successful Runs</div>
          <div className="stat-value">{stats?.successful_runs_24h || 0}</div>
          <div className="stat-change positive">
            <TrendingUp size={12} /> Last 24h
          </div>
        </div>

        <div className="stat-card" data-testid="stat-automl">
          <div className="stat-icon-wrapper automl">
            <Brain size={20} />
          </div>
          <div className="stat-label">AutoML Runs</div>
          <div className="stat-value">{stats?.automl_runs || 0}</div>
          <div className="stat-change positive">
            <Zap size={12} /> Automated
          </div>
        </div>

        {/* Pipeline Runs Chart */}
        <div className="chart-card" data-testid="chart-pipeline-runs">
          <h3 className="chart-title">
            <BarChart3 size={18} />
            Pipeline Runs (Last 7 Days)
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={metrics?.pipeline_runs || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="date" stroke="#a1a1aa" fontSize={12} />
              <YAxis stroke="#a1a1aa" fontSize={12} />
              <Tooltip 
                contentStyle={{ background: '#121212', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px' }}
                labelStyle={{ color: '#fafafa' }}
              />
              <Bar dataKey="success" fill="#10b981" radius={[4, 4, 0, 0]} name="Success" />
              <Bar dataKey="failed" fill="#ef4444" radius={[4, 4, 0, 0]} name="Failed" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Accuracy Chart */}
        <div className="chart-card" data-testid="chart-model-accuracy">
          <h3 className="chart-title">
            <Target size={18} />
            Model Accuracy Trend
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics?.model_accuracy || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="version" stroke="#a1a1aa" fontSize={12} />
              <YAxis stroke="#a1a1aa" fontSize={12} domain={[0.8, 1]} />
              <Tooltip 
                contentStyle={{ background: '#121212', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px' }}
                labelStyle={{ color: '#fafafa' }}
                formatter={(value) => [(value * 100).toFixed(1) + '%', 'Accuracy']}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={2} dot={{ fill: '#2563eb', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Runs */}
        <div className="list-card" data-testid="recent-runs-list">
          <div className="list-header">
            <h3 className="list-title">Recent Runs</h3>
            <ChevronRight size={16} className="text-muted-foreground" />
          </div>
          <div className="list-content">
            {recentRuns.length === 0 ? (
              <div className="empty-state">
                <p>No recent runs</p>
              </div>
            ) : (
              recentRuns.map(run => (
                <div key={run.id} className="list-item">
                  <StatusBadge status={run.status} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, fontSize: '0.875rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {run.pipeline_name}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                      {run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s` : 'Running...'}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Data Quality */}
        <div className="list-card" data-testid="data-quality-card">
          <div className="list-header">
            <h3 className="list-title">Data Quality Metrics</h3>
          </div>
          <div className="metrics-grid">
            {metrics?.data_quality && Object.entries(metrics.data_quality).map(([key, value]) => (
              <div key={key} className="metric-item">
                <div className="metric-label">{key}</div>
                <div className="metric-value">{value}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Validation Status */}
        <div className="list-card" data-testid="validation-status-card">
          <div className="list-header">
            <h3 className="list-title">Validation Status</h3>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Passed', value: stats?.data_validations_passed || 0 },
                    { name: 'Failed', value: stats?.data_validations_failed || 0 }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={70}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {[0, 1].map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Pipelines Page
// ============================================================================

const PipelinesPage = ({ lastMessage }) => {
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPipeline, setSelectedPipeline] = useState(null);
  const [runs, setRuns] = useState([]);

  const loadPipelines = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get('/api/pipelines');
      setPipelines(data);
    } catch (err) {
      console.error('Failed to load pipelines:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadPipelineRuns = useCallback(async (pipelineId) => {
    try {
      const data = await api.get(`/api/pipelines/${pipelineId}/runs`);
      setRuns(data);
    } catch (err) {
      console.error('Failed to load runs:', err);
    }
  }, []);

  useEffect(() => {
    loadPipelines();
  }, [loadPipelines]);

  useEffect(() => {
    if (selectedPipeline) {
      loadPipelineRuns(selectedPipeline.id);
    }
  }, [selectedPipeline, loadPipelineRuns]);

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'pipeline_completed' || lastMessage.type === 'pipeline_failed' || lastMessage.type === 'pipeline_step') {
        loadPipelines();
        if (selectedPipeline && lastMessage.data?.pipeline_id === selectedPipeline.id) {
          loadPipelineRuns(selectedPipeline.id);
        }
      }
    }
  }, [lastMessage, loadPipelines, selectedPipeline, loadPipelineRuns]);

  const runPipeline = async (pipelineId) => {
    try {
      await api.post(`/api/pipelines/${pipelineId}/run`);
      loadPipelines();
      if (selectedPipeline?.id === pipelineId) {
        loadPipelineRuns(pipelineId);
      }
    } catch (err) {
      console.error('Failed to run pipeline:', err);
    }
  };

  const deletePipeline = async (pipelineId) => {
    try {
      await api.delete(`/api/pipelines/${pipelineId}`);
      loadPipelines();
      if (selectedPipeline?.id === pipelineId) {
        setSelectedPipeline(null);
      }
    } catch (err) {
      console.error('Failed to delete pipeline:', err);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Pipelines</h1>
        <p className="page-description">Manage and monitor your ETL pipelines with real-time updates</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedPipeline ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-container" data-testid="pipelines-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Steps</th>
                <th>Runs</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : pipelines.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No pipelines found
                  </td>
                </tr>
              ) : (
                pipelines.map(pipeline => (
                  <tr key={pipeline.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedPipeline(pipeline)}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{pipeline.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{pipeline.description}</div>
                    </td>
                    <td><StatusBadge status={pipeline.status} /></td>
                    <td>{pipeline.steps?.length || 0}</td>
                    <td>{pipeline.run_count}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button 
                          className="btn-icon" 
                          data-testid={`run-pipeline-${pipeline.id}`}
                          onClick={() => runPipeline(pipeline.id)}
                          title="Run Pipeline"
                        >
                          <Play size={16} />
                        </button>
                        <button 
                          className="btn-icon" 
                          data-testid={`view-pipeline-${pipeline.id}`}
                          onClick={() => setSelectedPipeline(pipeline)}
                          title="View Details"
                        >
                          <Eye size={16} />
                        </button>
                        <button 
                          className="btn-icon" 
                          data-testid={`delete-pipeline-${pipeline.id}`}
                          onClick={() => deletePipeline(pipeline.id)}
                          title="Delete Pipeline"
                          style={{ color: '#ef4444' }}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {selectedPipeline && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-container" style={{ padding: '1.5rem' }} data-testid="pipeline-details">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{selectedPipeline.name}</h3>
                  <p style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>{selectedPipeline.description}</p>
                </div>
                <StatusBadge status={selectedPipeline.status} />
              </div>
              
              <div style={{ marginTop: '1.5rem' }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>Pipeline Steps</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {selectedPipeline.steps?.map((step, idx) => (
                    <div key={step.id} style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '0.75rem',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.02)',
                      borderRadius: '6px',
                      borderLeft: '3px solid #2563eb'
                    }}>
                      <div style={{ 
                        width: '24px', 
                        height: '24px', 
                        borderRadius: '50%', 
                        background: '#2563eb', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        fontWeight: 600
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>{step.name}</div>
                        <div style={{ fontSize: '0.75rem', color: '#a1a1aa', textTransform: 'capitalize' }}>{step.type}</div>
                      </div>
                      <span className={`status-badge ${step.type}`}>{step.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="table-container" data-testid="pipeline-runs">
              <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Run History</h4>
              </div>
              <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                {runs.length === 0 ? (
                  <div style={{ padding: '2rem', textAlign: 'center', color: '#a1a1aa' }}>No runs yet</div>
                ) : (
                  runs.map(run => (
                    <div key={run.id} style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between',
                      padding: '0.75rem 1rem',
                      borderBottom: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <StatusBadge status={run.status} />
                        <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                          {run.steps_completed}/{run.total_steps} steps
                        </span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                        {run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s` : 'Running...'}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Experiments Page
// ============================================================================

const ExperimentsPage = ({ lastMessage }) => {
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('experiments');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newExperiment, setNewExperiment] = useState({
    name: '',
    description: '',
    algorithm: 'RandomForest',
    n_estimators: 100,
    max_depth: 10
  });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [expData, modelData] = await Promise.all([
        api.get('/api/experiments'),
        api.get('/api/models')
      ]);
      setExperiments(expData);
      setModels(modelData);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage?.type === 'experiment_completed') {
      loadData();
    }
  }, [lastMessage, loadData]);

  const createExperiment = async () => {
    try {
      await api.post('/api/experiments', {
        name: newExperiment.name,
        description: newExperiment.description,
        algorithm: newExperiment.algorithm,
        parameters: {
          n_estimators: newExperiment.n_estimators,
          max_depth: newExperiment.max_depth
        }
      });
      setShowCreateModal(false);
      setNewExperiment({ name: '', description: '', algorithm: 'RandomForest', n_estimators: 100, max_depth: 10 });
      loadData();
    } catch (err) {
      console.error('Failed to create experiment:', err);
    }
  };

  const deleteExperiment = async (expId) => {
    try {
      await api.delete(`/api/experiments/${expId}`);
      loadData();
    } catch (err) {
      console.error('Failed to delete experiment:', err);
    }
  };

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 className="page-title">Experiments</h1>
            <p className="page-description">Track ML experiments and model versions</p>
          </div>
          <button className="btn btn-primary" data-testid="create-experiment-btn" onClick={() => setShowCreateModal(true)}>
            <Plus size={16} />
            New Experiment
          </button>
        </div>
      </div>

      {showCreateModal && (
        <div className="modal-overlay" data-testid="experiment-modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal" data-testid="experiment-modal" onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Create New Experiment</h3>
              <button 
                className="btn-icon" 
                data-testid="experiment-modal-close"
                onClick={() => setShowCreateModal(false)}
                style={{ padding: '0.25rem' }}
              >
                <XCircle size={20} />
              </button>
            </div>
            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                className="input"
                value={newExperiment.name}
                onChange={e => setNewExperiment({...newExperiment, name: e.target.value})}
                placeholder="Experiment name"
              />
            </div>
            <div className="form-group">
              <label>Description</label>
              <input
                type="text"
                className="input"
                value={newExperiment.description}
                onChange={e => setNewExperiment({...newExperiment, description: e.target.value})}
                placeholder="Description"
              />
            </div>
            <div className="form-group">
              <label>Algorithm</label>
              <select
                className="input"
                value={newExperiment.algorithm}
                onChange={e => setNewExperiment({...newExperiment, algorithm: e.target.value})}
              >
                <option value="RandomForest">Random Forest</option>
                <option value="GradientBoosting">Gradient Boosting</option>
                <option value="LogisticRegression">Logistic Regression</option>
                <option value="SVM">SVM</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label>N Estimators</label>
                <input
                  type="number"
                  className="input"
                  value={newExperiment.n_estimators}
                  onChange={e => setNewExperiment({...newExperiment, n_estimators: parseInt(e.target.value)})}
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>Max Depth</label>
                <input
                  type="number"
                  className="input"
                  value={newExperiment.max_depth}
                  onChange={e => setNewExperiment({...newExperiment, max_depth: parseInt(e.target.value)})}
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
              <button className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={createExperiment} disabled={!newExperiment.name}>
                <Zap size={16} />
                Run Experiment
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'experiments' ? 'active' : ''}`}
          data-testid="tab-experiments"
          onClick={() => setActiveTab('experiments')}
        >
          Experiments
        </button>
        <button 
          className={`tab ${activeTab === 'models' ? 'active' : ''}`}
          data-testid="tab-models"
          onClick={() => setActiveTab('models')}
        >
          Models
        </button>
      </div>

      {activeTab === 'experiments' && (
        <div className="table-container" data-testid="experiments-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Algorithm</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>Model Version</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : experiments.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No experiments found
                  </td>
                </tr>
              ) : (
                experiments.map(exp => (
                  <tr key={exp.id}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{exp.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{exp.id}</div>
                    </td>
                    <td>{exp.algorithm || exp.parameters?.algorithm || '-'}</td>
                    <td><StatusBadge status={exp.status === 'completed' ? 'success' : exp.status} /></td>
                    <td>
                      {exp.metrics?.accuracy ? (
                        <span style={{ color: '#10b981', fontWeight: 500 }}>
                          {(exp.metrics.accuracy * 100).toFixed(2)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td>
                      {exp.metrics?.f1_score ? (
                        <span>{(exp.metrics.f1_score * 100).toFixed(2)}%</span>
                      ) : '-'}
                    </td>
                    <td>
                      {exp.model_version ? (
                        <span className="status-badge success">{exp.model_version}</span>
                      ) : '-'}
                    </td>
                    <td>
                      <button 
                        className="btn-icon"
                        data-testid={`delete-experiment-${exp.id}`}
                        onClick={() => deleteExperiment(exp.id)}
                        style={{ color: '#ef4444' }}
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="table-container" data-testid="models-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : models.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No models found
                  </td>
                </tr>
              ) : (
                models.map(model => (
                  <tr key={model.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Box size={16} className="text-primary" />
                        <span style={{ fontWeight: 500 }}>{model.name}</span>
                      </div>
                    </td>
                    <td>
                      <span className="status-badge success">{model.version}</span>
                    </td>
                    <td>{model.algorithm}</td>
                    <td>
                      <span style={{ color: '#10b981', fontWeight: 500 }}>
                        {model.metrics?.accuracy ? (model.metrics.accuracy * 100).toFixed(2) + '%' : '-'}
                      </span>
                    </td>
                    <td>{model.metrics?.precision ? (model.metrics.precision * 100).toFixed(2) + '%' : '-'}</td>
                    <td>{model.metrics?.recall ? (model.metrics.recall * 100).toFixed(2) + '%' : '-'}</td>
                    <td><StatusBadge status="success" /></td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// AutoML Page
// ============================================================================

const AutoMLPage = ({ lastMessage }) => {
  const [automlRuns, setAutomlRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [progress, setProgress] = useState(null);
  const [newAutoML, setNewAutoML] = useState({
    experiment_name: '',
    description: '',
    algorithms: ['RandomForest', 'GradientBoosting', 'LogisticRegression'],
    cv_folds: 5,
    max_trials: 20
  });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get('/api/automl/runs');
      setAutomlRuns(data);
    } catch (err) {
      console.error('Failed to load AutoML runs:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle real-time updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'automl_progress') {
        setProgress(lastMessage.data);
      } else if (lastMessage.type === 'automl_completed') {
        loadData();
        setProgress(null);
      }
    }
  }, [lastMessage, loadData]);

  const runAutoML = async () => {
    try {
      await api.post('/api/automl/run', newAutoML);
      setShowCreateModal(false);
      setNewAutoML({
        experiment_name: '',
        description: '',
        algorithms: ['RandomForest', 'GradientBoosting', 'LogisticRegression'],
        cv_folds: 5,
        max_trials: 20
      });
      loadData();
    } catch (err) {
      console.error('Failed to run AutoML:', err);
    }
  };

  const toggleAlgorithm = (algo) => {
    const current = newAutoML.algorithms;
    if (current.includes(algo)) {
      setNewAutoML({...newAutoML, algorithms: current.filter(a => a !== algo)});
    } else {
      setNewAutoML({...newAutoML, algorithms: [...current, algo]});
    }
  };

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1 className="page-title">
              <Brain size={28} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
              AutoML
            </h1>
            <p className="page-description">Automated machine learning with hyperparameter optimization</p>
          </div>
          <button className="btn btn-primary" data-testid="run-automl-btn" onClick={() => setShowCreateModal(true)}>
            <Sparkles size={16} />
            New AutoML Run
          </button>
        </div>
      </div>

      {progress && (
        <div className="automl-progress-banner" data-testid="automl-progress">
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <Loader2 size={20} className="animate-spin" />
            <div>
              <div style={{ fontWeight: 500 }}>AutoML in progress...</div>
              <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                Testing {progress.algorithm} | Progress: {progress.progress?.toFixed(0)}% | Best Score: {progress.current_best_score?.toFixed(4)}
              </div>
            </div>
          </div>
          <div className="progress-bar" style={{ marginTop: '0.75rem' }}>
            <div className="progress-fill success" style={{ width: `${progress.progress || 0}%` }} />
          </div>
        </div>
      )}

      {showCreateModal && (
        <div className="modal-overlay" data-testid="automl-modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal modal-lg" data-testid="automl-modal" onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
                <Brain size={24} />
                Configure AutoML Run
              </h3>
              <button 
                className="btn-icon" 
                data-testid="automl-modal-close"
                onClick={() => setShowCreateModal(false)}
                style={{ padding: '0.25rem' }}
              >
                <XCircle size={20} />
              </button>
            </div>
            <div className="form-group">
              <label>Experiment Name</label>
              <input
                type="text"
                className="input"
                value={newAutoML.experiment_name}
                onChange={e => setNewAutoML({...newAutoML, experiment_name: e.target.value})}
                placeholder="e.g., Activity Recognition AutoML"
              />
            </div>
            <div className="form-group">
              <label>Description</label>
              <input
                type="text"
                className="input"
                value={newAutoML.description}
                onChange={e => setNewAutoML({...newAutoML, description: e.target.value})}
                placeholder="Description of the AutoML run"
              />
            </div>
            <div className="form-group">
              <label>Algorithms to Test</label>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'].map(algo => (
                  <button
                    key={algo}
                    className={`algorithm-chip ${newAutoML.algorithms.includes(algo) ? 'active' : ''}`}
                    onClick={() => toggleAlgorithm(algo)}
                  >
                    {newAutoML.algorithms.includes(algo) && <CheckCircle2 size={14} />}
                    {algo}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <div className="form-group" style={{ flex: 1 }}>
                <label>Cross-Validation Folds</label>
                <input
                  type="number"
                  className="input"
                  value={newAutoML.cv_folds}
                  onChange={e => setNewAutoML({...newAutoML, cv_folds: parseInt(e.target.value)})}
                  min={2}
                  max={10}
                />
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>Max Trials</label>
                <input
                  type="number"
                  className="input"
                  value={newAutoML.max_trials}
                  onChange={e => setNewAutoML({...newAutoML, max_trials: parseInt(e.target.value)})}
                  min={5}
                  max={100}
                />
              </div>
            </div>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', marginTop: '1.5rem' }}>
              <button className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>Cancel</button>
              <button 
                className="btn btn-primary" 
                onClick={runAutoML} 
                disabled={!newAutoML.experiment_name || newAutoML.algorithms.length === 0}
              >
                <Zap size={16} />
                Start AutoML
              </button>
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: selectedRun ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-container" data-testid="automl-runs-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Algorithms</th>
                <th>Best Score</th>
                <th>Trials</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : automlRuns.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No AutoML runs found. Click "New AutoML Run" to start one.
                  </td>
                </tr>
              ) : (
                automlRuns.map(run => (
                  <tr key={run.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedRun(run)}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{run.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{run.id}</div>
                    </td>
                    <td><StatusBadge status={run.status === 'completed' ? 'success' : run.status} /></td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                        {run.algorithms?.slice(0, 2).map(algo => (
                          <span key={algo} className="status-badge idle" style={{ fontSize: '0.65rem' }}>{algo}</span>
                        ))}
                        {run.algorithms?.length > 2 && <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>+{run.algorithms.length - 2}</span>}
                      </div>
                    </td>
                    <td>
                      {run.best_score ? (
                        <span style={{ color: '#10b981', fontWeight: 600 }}>
                          {(run.best_score * 100).toFixed(2)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td>{run.total_trials || '-'}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <button className="btn-icon" onClick={() => setSelectedRun(run)}>
                        <Eye size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {selectedRun && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-container" style={{ padding: '1.5rem' }} data-testid="automl-details">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{selectedRun.name}</h3>
                  <p style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{selectedRun.description}</p>
                </div>
                <StatusBadge status={selectedRun.status === 'completed' ? 'success' : selectedRun.status} />
              </div>

              {selectedRun.best_algorithm && (
                <div className="best-model-card">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <Target size={18} color="#10b981" />
                    <span style={{ fontWeight: 600, color: '#10b981' }}>Best Model</span>
                  </div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{selectedRun.best_algorithm}</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: '#10b981' }}>
                    {(selectedRun.best_score * 100).toFixed(2)}% accuracy
                  </div>
                </div>
              )}

              <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                <div className="metric-item">
                  <div className="metric-label">Total Trials</div>
                  <div className="metric-value">{selectedRun.total_trials || 0}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">CV Folds</div>
                  <div className="metric-value">{selectedRun.cv_folds}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Algorithms</div>
                  <div className="metric-value">{selectedRun.algorithms?.length || 0}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Metric</div>
                  <div className="metric-value">{selectedRun.scoring_metric}</div>
                </div>
              </div>
            </div>

            {selectedRun.results?.length > 0 && (
              <div className="table-container" data-testid="automl-results">
                <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Algorithm Results</h4>
                </div>
                <div style={{ padding: '1rem' }}>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={selectedRun.results.map(r => ({ name: r.algorithm, score: r.test_score * 100 }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="#a1a1aa" fontSize={11} />
                      <YAxis stroke="#a1a1aa" fontSize={11} domain={[0, 100]} />
                      <Tooltip 
                        contentStyle={{ background: '#121212', border: '1px solid rgba(255,255,255,0.1)' }}
                        formatter={(value) => [value.toFixed(2) + '%', 'Accuracy']}
                      />
                      <Bar dataKey="score" fill="#2563eb" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Data Quality / Validations Page
// ============================================================================

const ValidationsPage = () => {
  const [validations, setValidations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedValidation, setSelectedValidation] = useState(null);

  const loadValidations = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get('/api/validations');
      setValidations(data);
    } catch (err) {
      console.error('Failed to load validations:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadValidations();
  }, [loadValidations]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Data Quality</h1>
        <p className="page-description">Monitor data validation results and quality metrics</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedValidation ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-container" data-testid="validations-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Dataset</th>
                <th>Status</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : validations.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No validations found
                  </td>
                </tr>
              ) : (
                validations.map(val => (
                  <tr key={val.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedValidation(val)}>
                    <td style={{ fontWeight: 500 }}>{val.name}</td>
                    <td style={{ fontSize: '0.75rem', color: '#a1a1aa', fontFamily: 'JetBrains Mono, monospace' }}>
                      {val.dataset_path}
                    </td>
                    <td><StatusBadge status={val.status === 'passed' ? 'success' : 'failed'} /></td>
                    <td style={{ color: '#10b981' }}>{val.rules_passed}</td>
                    <td style={{ color: val.rules_failed > 0 ? '#ef4444' : '#a1a1aa' }}>{val.rules_failed}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <button 
                        className="btn-icon"
                        data-testid={`view-validation-${val.id}`}
                        onClick={() => setSelectedValidation(val)}
                      >
                        <Eye size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {selectedValidation && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-container" style={{ padding: '1.5rem' }} data-testid="validation-details">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{selectedValidation.name}</h3>
                  <p style={{ fontSize: '0.75rem', color: '#a1a1aa', fontFamily: 'JetBrains Mono' }}>
                    {selectedValidation.dataset_path}
                  </p>
                </div>
                <StatusBadge status={selectedValidation.status === 'passed' ? 'success' : 'failed'} />
              </div>

              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-label">Total Rows</div>
                  <div className="metric-value">{selectedValidation.profile?.total_rows?.toLocaleString()}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Columns</div>
                  <div className="metric-value">{selectedValidation.profile?.total_columns}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Missing Cells</div>
                  <div className="metric-value">{selectedValidation.profile?.missing_cells}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Size</div>
                  <div className="metric-value">{selectedValidation.profile?.memory_size_mb} MB</div>
                </div>
              </div>

              <div style={{ marginTop: '1.5rem' }}>
                <div style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.875rem' }}>Validation Progress</span>
                  <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                    {selectedValidation.rules_passed}/{selectedValidation.total_rules} rules passed
                  </span>
                </div>
                <div className="progress-bar">
                  <div 
                    className={`progress-fill ${selectedValidation.rules_failed > 0 ? 'error' : 'success'}`}
                    style={{ width: `${(selectedValidation.rules_passed / selectedValidation.total_rules) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {selectedValidation.issues?.length > 0 && (
              <div className="table-container" data-testid="validation-issues">
                <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle size={16} className="text-warning" />
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Issues ({selectedValidation.issues.length})</h4>
                </div>
                <div style={{ maxHeight: '250px', overflow: 'auto' }}>
                  {selectedValidation.issues.map((issue, idx) => (
                    <div key={idx} style={{ 
                      padding: '0.75rem 1rem',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.75rem'
                    }}>
                      <span className={`status-badge ${issue.severity === 'high' ? 'failed' : issue.severity === 'medium' ? 'pending' : 'idle'}`}>
                        {issue.severity}
                      </span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>{issue.type}</div>
                        <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                          {issue.affected_rows} affected rows
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Logs Page
// ============================================================================

const LogsPage = ({ lastMessage }) => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  const loadLogs = useCallback(async () => {
    try {
      setLoading(true);
      const endpoint = filter === 'all' ? '/api/logs' : `/api/logs?level=${filter.toUpperCase()}`;
      const data = await api.get(endpoint);
      setLogs(data);
    } catch (err) {
      console.error('Failed to load logs:', err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  // Handle real-time log updates
  useEffect(() => {
    if (lastMessage?.type === 'log') {
      setLogs(prev => [lastMessage.data, ...prev.slice(0, 99)]);
    }
  }, [lastMessage]);

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">Logs</h1>
            <p className="page-description">Real-time system logs and activity history</p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {['all', 'info', 'warning', 'error'].map(level => (
              <button
                key={level}
                className={`btn ${filter === level ? 'btn-primary' : 'btn-secondary'}`}
                data-testid={`filter-${level}`}
                onClick={() => setFilter(level)}
              >
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="log-viewer" data-testid="log-viewer">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
          </div>
        ) : logs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
            No logs found
          </div>
        ) : (
          logs.map(log => (
            <div key={log.id} className="log-entry">
              <span className="log-timestamp">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span className={`log-level ${log.level.toLowerCase()}`}>
                [{log.level}]
              </span>
              <span className="log-source">{log.source}</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Settings Page
// ============================================================================

const SettingsPage = () => {
  const [seeding, setSeeding] = useState(false);
  const [seedResult, setSeedResult] = useState(null);

  const seedDatabase = async () => {
    try {
      setSeeding(true);
      const result = await api.post('/api/seed');
      setSeedResult(result);
    } catch (err) {
      console.error('Failed to seed database:', err);
      setSeedResult({ error: err.message });
    } finally {
      setSeeding(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-description">Application configuration and administration</p>
      </div>

      <div style={{ display: 'grid', gap: '1.5rem', maxWidth: '600px' }}>
        <div className="table-container" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Database</h3>
          <p style={{ fontSize: '0.875rem', color: '#a1a1aa', marginBottom: '1rem' }}>
            Seed the database with sample data for testing and demonstration purposes.
          </p>
          <button 
            className="btn btn-primary"
            data-testid="seed-database-btn"
            onClick={seedDatabase}
            disabled={seeding}
          >
            {seeding ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Seeding...
              </>
            ) : (
              <>
                <Database size={16} />
                Seed Database
              </>
            )}
          </button>
          
          {seedResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: seedResult.error ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)', 
              borderRadius: '6px',
              fontSize: '0.875rem'
            }}>
              {seedResult.error ? (
                <span style={{ color: '#ef4444' }}>Error: {seedResult.error}</span>
              ) : (
                <div>
                  <div style={{ color: '#10b981', fontWeight: 500, marginBottom: '0.5rem' }}>
                    Database seeded successfully!
                  </div>
                  <div style={{ color: '#a1a1aa', fontSize: '0.75rem' }}>
                    Created: {seedResult.pipelines} pipelines, {seedResult.runs} runs, {seedResult.experiments} experiments, {seedResult.models} models, {seedResult.automl_runs} AutoML runs
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="table-container" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Features</h3>
          <div style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 0' }}>
                <CheckCircle2 size={16} color="#10b981" /> Real-time WebSocket monitoring
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 0' }}>
                <CheckCircle2 size={16} color="#10b981" /> AutoML with hyperparameter tuning
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 0' }}>
                <CheckCircle2 size={16} color="#10b981" /> sklearn model training
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 0' }}>
                <CheckCircle2 size={16} color="#10b981" /> Background pipeline execution
              </li>
              <li style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 0' }}>
                <CheckCircle2 size={16} color="#10b981" /> AWS S3 integration ready
              </li>
            </ul>
          </div>
        </div>

        <div className="table-container" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>About</h3>
          <div style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>
            <p style={{ marginBottom: '0.5rem' }}>ETL & ML Dashboard v2.0.0</p>
            <p>A comprehensive FAANG-level platform for managing ETL pipelines, tracking ML experiments, and automated machine learning.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main App Component
// ============================================================================

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const { isConnected, lastMessage, sendMessage } = useWebSocket();

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage lastMessage={lastMessage} />;
      case 'pipelines':
        return <PipelinesPage lastMessage={lastMessage} />;
      case 'experiments':
        return <ExperimentsPage lastMessage={lastMessage} />;
      case 'automl':
        return <AutoMLPage lastMessage={lastMessage} />;
      case 'validations':
        return <ValidationsPage />;
      case 'logs':
        return <LogsPage lastMessage={lastMessage} />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage lastMessage={lastMessage} />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activePage={activePage} setActivePage={setActivePage} isConnected={isConnected} />
      <main className="main-content" data-testid="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
