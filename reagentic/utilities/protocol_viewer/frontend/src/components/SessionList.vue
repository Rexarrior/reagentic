<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { format, isToday, isYesterday, parseISO } from 'date-fns'
import { useProtocolStore } from '../stores/protocol'
import type { SessionInfo } from '../types/protocol'
import { formatDuration } from '../types/protocol'

const store = useProtocolStore()

onMounted(() => {
  store.fetchSessions()
})

// Group sessions by date
const groupedSessions = computed(() => {
  const groups: Record<string, SessionInfo[]> = {}
  
  for (const session of store.sessions) {
    const date = parseISO(session.start_time)
    let label: string
    
    if (isToday(date)) {
      label = 'Today'
    } else if (isYesterday(date)) {
      label = 'Yesterday'
    } else {
      label = format(date, 'MMM d, yyyy')
    }
    
    if (!groups[label]) {
      groups[label] = []
    }
    groups[label].push(session)
  }
  
  return groups
})

function formatTime(isoString: string): string {
  return format(parseISO(isoString), 'HH:mm:ss')
}

function selectSession(session: SessionInfo) {
  store.fetchSession(session.id)
}

function isSelected(session: SessionInfo): boolean {
  return store.sessionInfo?.id === session.id
}
</script>

<template>
  <div class="session-list">
    <div class="session-list__header">
      <h2>Sessions</h2>
      <button class="refresh-btn" @click="store.fetchSessions" :disabled="store.loading">
        ↻
      </button>
    </div>
    
    <div v-if="store.loading && !store.sessions.length" class="session-list__loading">
      Loading sessions...
    </div>
    
    <div v-else-if="store.error" class="session-list__error">
      {{ store.error }}
    </div>
    
    <div v-else-if="!store.sessions.length" class="session-list__empty">
      No sessions found
    </div>
    
    <div v-else class="session-list__content">
      <div v-for="(sessions, date) in groupedSessions" :key="date" class="session-group">
        <div class="session-group__header">{{ date }}</div>
        
        <div
          v-for="session in sessions"
          :key="session.id"
          class="session-item"
          :class="{
            'session-item--selected': isSelected(session),
            'session-item--error': session.has_error
          }"
          @click="selectSession(session)"
        >
          <div class="session-item__main">
            <div class="session-item__name">
              {{ session.agent_name || 'Unknown Agent' }}
            </div>
            <div class="session-item__time">
              {{ formatTime(session.start_time) }}
            </div>
          </div>
          
          <div class="session-item__meta">
            <span class="session-item__events">
              {{ session.event_count }} events
            </span>
            <span v-if="session.duration_ms" class="session-item__duration">
              {{ formatDuration(session.duration_ms) }}
            </span>
            <span v-if="session.total_tokens" class="session-item__tokens">
              {{ session.total_tokens }} tokens
            </span>
          </div>
          
          <div v-if="session.has_error" class="session-item__error-badge">
            ✕
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.session-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-subtle);
}

.session-list__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.session-list__header h2 {
  font-size: 14px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

.refresh-btn {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.refresh-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.session-list__content {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-list__loading,
.session-list__error,
.session-list__empty {
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
}

.session-list__error {
  color: var(--accent-red);
}

.session-group {
  margin-bottom: 16px;
}

.session-group__header {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  padding: 8px 8px 4px;
}

.session-item {
  position: relative;
  padding: 12px;
  margin: 4px 0;
  background: var(--bg-tertiary);
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.2s;
}

.session-item:hover {
  background: var(--bg-elevated);
  border-color: var(--border-color);
}

.session-item--selected {
  background: var(--bg-elevated);
  border-color: var(--accent-cyan);
  box-shadow: 0 0 0 1px rgba(0, 217, 255, 0.2);
}

.session-item--error {
  border-left: 3px solid var(--accent-red);
}

.session-item__main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.session-item__name {
  font-weight: 500;
  font-size: 13px;
  color: var(--text-primary);
}

.session-item__time {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}

.session-item__meta {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--text-secondary);
}

.session-item__error-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-red);
  color: white;
  border-radius: 50%;
  font-size: 10px;
  font-weight: bold;
}
</style>
