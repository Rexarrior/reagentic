/**
 * Protocol types matching the backend API.
 */

export interface SessionInfo {
  id: string
  agent_name: string | null
  start_time: string
  end_time: string | null
  duration_ms: number | null
  event_count: number
  has_error: boolean
  total_tokens: number | null
}

export interface ProtocolEntry {
  id: string
  timestamp: string
  event_type: EventType
  session_id: string | null
  agent_name: string | null
  agent_id: string | null
  tool_name: string | null
  input_data: unknown
  output_data: unknown
  system_prompt: string | null
  tokens_used: number | null
  duration_ms: number | null
  intermediate_steps: unknown[] | null
  trace_id: string | null
  span_id: string | null
  metadata: Record<string, unknown> | null
  error: string | null
}

export type EventType =
  | 'agent_start'
  | 'agent_end'
  | 'llm_start'
  | 'llm_end'
  | 'tool_start'
  | 'tool_end'
  | 'handoff'
  | 'trace_start'
  | 'trace_end'
  | 'span_start'
  | 'span_end'

export interface SessionDetail {
  session: SessionInfo
  entries: ProtocolEntry[]
}

export interface StatsResponse {
  total_sessions: number
  total_events: number
  total_tokens: number
  total_duration_ms: number
  event_type_counts: Record<string, number>
}

export interface GraphNode {
  id: string
  type: string
  data: {
    label: string
    event_type: EventType
    entry_id: string
    timestamp: string
    duration_ms: number | null
    tokens: number | null
    has_error: boolean
  }
  position: { x: number; y: number }
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  animated: boolean
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

// Helper to get color for event type
export function getEventColor(eventType: EventType): string {
  switch (eventType) {
    case 'agent_start':
    case 'agent_end':
      return 'var(--accent-cyan)'
    case 'llm_start':
    case 'llm_end':
      return 'var(--accent-purple)'
    case 'tool_start':
    case 'tool_end':
      return 'var(--accent-orange)'
    case 'handoff':
      return 'var(--accent-pink)'
    case 'trace_start':
    case 'trace_end':
    case 'span_start':
    case 'span_end':
      return 'var(--accent-green)'
    default:
      return 'var(--text-muted)'
  }
}

// Helper to get icon for event type
export function getEventIcon(eventType: EventType): string {
  switch (eventType) {
    case 'agent_start':
      return '▶'
    case 'agent_end':
      return '⏹'
    case 'llm_start':
      return '🤖'
    case 'llm_end':
      return '💬'
    case 'tool_start':
      return '🔧'
    case 'tool_end':
      return '✓'
    case 'handoff':
      return '↪'
    default:
      return '●'
  }
}

// Helper to format duration
export function formatDuration(ms: number | null): string {
  if (ms === null) return '-'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}
