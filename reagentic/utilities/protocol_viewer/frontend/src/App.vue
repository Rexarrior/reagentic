<script setup lang="ts">
import { onMounted } from 'vue'
import { useProtocolStore } from './stores/protocol'
import SessionList from './components/SessionList.vue'
import AgentGraph from './components/AgentGraph.vue'
import EventTimeline from './components/EventTimeline.vue'
import DetailPanel from './components/DetailPanel.vue'

const store = useProtocolStore()

onMounted(() => {
  store.fetchStats()
})
</script>

<template>
  <div class="app">
    <header class="app-header">
      <div class="header-left">
        <h1 class="logo">
          <span class="logo-icon">◈</span>
          Protocol Viewer
        </h1>
      </div>
      
      <div class="header-center">
        <div v-if="store.sessionInfo" class="session-badge">
          <span class="session-name">{{ store.sessionInfo.agent_name || 'Session' }}</span>
          <span class="session-events">{{ store.sessionInfo.event_count }} events</span>
        </div>
      </div>
      
      <div class="header-right">
        <div v-if="store.stats" class="stats">
          <div class="stat">
            <span class="stat-value">{{ store.stats.total_sessions }}</span>
            <span class="stat-label">sessions</span>
          </div>
          <div class="stat">
            <span class="stat-value">{{ store.stats.total_events }}</span>
            <span class="stat-label">events</span>
          </div>
          <div class="stat">
            <span class="stat-value">{{ store.stats.total_tokens.toLocaleString() }}</span>
            <span class="stat-label">tokens</span>
          </div>
        </div>
      </div>
    </header>
    
    <main class="app-main">
      <aside class="sidebar">
        <SessionList />
      </aside>
      
      <section class="content">
        <div class="graph-area">
          <AgentGraph />
        </div>
        <div class="timeline-area">
          <EventTimeline />
        </div>
      </section>
      
      <aside class="detail-sidebar">
        <DetailPanel />
      </aside>
    </main>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-primary);
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 56px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-subtle);
  flex-shrink: 0;
}

.header-left {
  flex: 1;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.logo-icon {
  color: var(--accent-cyan);
  font-size: 24px;
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
}

.session-badge {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
}

.session-name {
  font-weight: 500;
  color: var(--text-primary);
}

.session-events {
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-mono);
}

.header-right {
  flex: 1;
  display: flex;
  justify-content: flex-end;
}

.stats {
  display: flex;
  gap: 24px;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 16px;
  font-weight: 600;
  color: var(--accent-cyan);
  font-family: var(--font-mono);
}

.stat-label {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
}

.app-main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 280px;
  flex-shrink: 0;
}

.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.graph-area {
  flex: 1;
  min-height: 300px;
}

.timeline-area {
  height: 220px;
  flex-shrink: 0;
}

.detail-sidebar {
  width: 360px;
  flex-shrink: 0;
}

/* Responsive adjustments */
@media (max-width: 1400px) {
  .detail-sidebar {
    width: 300px;
  }
}

@media (max-width: 1200px) {
  .sidebar {
    width: 240px;
  }
  
  .stats {
    gap: 16px;
  }
}
</style>
