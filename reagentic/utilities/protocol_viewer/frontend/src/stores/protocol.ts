/**
 * Pinia store for protocol data management.
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { SessionInfo, SessionDetail, ProtocolEntry, StatsResponse, GraphData } from '../types/protocol'

const API_BASE = '/api'

export const useProtocolStore = defineStore('protocol', () => {
  // State
  const sessions = ref<SessionInfo[]>([])
  const currentSession = ref<SessionDetail | null>(null)
  const currentEntry = ref<ProtocolEntry | null>(null)
  const graphData = ref<GraphData | null>(null)
  const stats = ref<StatsResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const selectedEntryId = ref<string | null>(null)

  // Computed
  const hasSession = computed(() => currentSession.value !== null)
  const entries = computed(() => currentSession.value?.entries ?? [])
  const sessionInfo = computed(() => currentSession.value?.session ?? null)

  // Actions
  async function fetchSessions() {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(`${API_BASE}/sessions`)
      if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.statusText}`)
      sessions.value = await res.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
    } finally {
      loading.value = false
    }
  }

  async function fetchSession(sessionId: string) {
    loading.value = true
    error.value = null
    try {
      const [sessionRes, graphRes] = await Promise.all([
        fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}`),
        fetch(`${API_BASE}/graph/${encodeURIComponent(sessionId)}`),
      ])
      
      if (!sessionRes.ok) throw new Error(`Failed to fetch session: ${sessionRes.statusText}`)
      if (!graphRes.ok) throw new Error(`Failed to fetch graph: ${graphRes.statusText}`)
      
      currentSession.value = await sessionRes.json()
      graphData.value = await graphRes.json()
      
      // Select first entry by default
      if (currentSession.value?.entries.length) {
        selectEntry(currentSession.value.entries[0].id)
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
    } finally {
      loading.value = false
    }
  }

  async function fetchStats() {
    try {
      const res = await fetch(`${API_BASE}/stats`)
      if (!res.ok) throw new Error(`Failed to fetch stats: ${res.statusText}`)
      stats.value = await res.json()
    } catch (e) {
      console.error('Failed to fetch stats:', e)
    }
  }

  function selectEntry(entryId: string | null) {
    selectedEntryId.value = entryId
    if (entryId && currentSession.value) {
      currentEntry.value = currentSession.value.entries.find(e => e.id === entryId) ?? null
    } else {
      currentEntry.value = null
    }
  }

  function clearSession() {
    currentSession.value = null
    currentEntry.value = null
    graphData.value = null
    selectedEntryId.value = null
  }

  // Navigate to next/previous entry
  function selectNextEntry() {
    if (!currentSession.value || !selectedEntryId.value) return
    const entries = currentSession.value.entries
    const idx = entries.findIndex(e => e.id === selectedEntryId.value)
    if (idx < entries.length - 1) {
      selectEntry(entries[idx + 1].id)
    }
  }

  function selectPrevEntry() {
    if (!currentSession.value || !selectedEntryId.value) return
    const entries = currentSession.value.entries
    const idx = entries.findIndex(e => e.id === selectedEntryId.value)
    if (idx > 0) {
      selectEntry(entries[idx - 1].id)
    }
  }

  return {
    // State
    sessions,
    currentSession,
    currentEntry,
    graphData,
    stats,
    loading,
    error,
    selectedEntryId,
    
    // Computed
    hasSession,
    entries,
    sessionInfo,
    
    // Actions
    fetchSessions,
    fetchSession,
    fetchStats,
    selectEntry,
    clearSession,
    selectNextEntry,
    selectPrevEntry,
  }
})
