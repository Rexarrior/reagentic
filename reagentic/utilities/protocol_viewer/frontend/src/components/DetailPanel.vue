<script setup lang="ts">
import { computed } from 'vue'
import { format, parseISO } from 'date-fns'
import { useProtocolStore } from '../stores/protocol'
import { getEventColor, getEventIcon, formatDuration } from '../types/protocol'
import type { EventType } from '../types/protocol'

const store = useProtocolStore()

const entry = computed(() => store.currentEntry)

function formatTimestamp(isoString: string): string {
  return format(parseISO(isoString), 'yyyy-MM-dd HH:mm:ss.SSS')
}

function formatJson(data: unknown): string {
  if (data === null || data === undefined) return 'null'
  try {
    return JSON.stringify(data, null, 2)
  } catch {
    return String(data)
  }
}

function isStringData(data: unknown): boolean {
  return typeof data === 'string'
}
</script>

<template>
  <div class="detail-panel">
    <div v-if="!entry" class="detail-panel__empty">
      <div class="empty-icon">📋</div>
      <div class="empty-text">Select an event to view details</div>
    </div>
    
    <template v-else>
      <div class="detail-panel__header">
        <div 
          class="header__badge"
          :style="{ backgroundColor: getEventColor(entry.event_type as EventType) }"
        >
          <span class="header__icon">{{ getEventIcon(entry.event_type as EventType) }}</span>
          <span class="header__type">{{ entry.event_type }}</span>
        </div>
        
        <div v-if="entry.error" class="header__error-badge">
          Error
        </div>
      </div>
      
      <div class="detail-panel__content">
        <!-- Metadata section -->
        <section class="detail-section">
          <h3 class="section-title">Metadata</h3>
          
          <div class="metadata-grid">
            <div class="metadata-item">
              <span class="metadata-label">Timestamp</span>
              <span class="metadata-value mono">{{ formatTimestamp(entry.timestamp) }}</span>
            </div>
            
            <div v-if="entry.agent_name" class="metadata-item">
              <span class="metadata-label">Agent</span>
              <span class="metadata-value">{{ entry.agent_name }}</span>
            </div>
            
            <div v-if="entry.tool_name" class="metadata-item">
              <span class="metadata-label">Tool</span>
              <span class="metadata-value">{{ entry.tool_name }}</span>
            </div>
            
            <div v-if="entry.duration_ms !== null" class="metadata-item">
              <span class="metadata-label">Duration</span>
              <span class="metadata-value highlight">{{ formatDuration(entry.duration_ms) }}</span>
            </div>
            
            <div v-if="entry.tokens_used !== null" class="metadata-item">
              <span class="metadata-label">Tokens</span>
              <span class="metadata-value highlight">{{ entry.tokens_used }}</span>
            </div>
            
            <div v-if="entry.session_id" class="metadata-item full-width">
              <span class="metadata-label">Session ID</span>
              <span class="metadata-value mono small">{{ entry.session_id }}</span>
            </div>
            
            <div class="metadata-item full-width">
              <span class="metadata-label">Entry ID</span>
              <span class="metadata-value mono small">{{ entry.id }}</span>
            </div>
          </div>
        </section>
        
        <!-- Error section -->
        <section v-if="entry.error" class="detail-section error-section">
          <h3 class="section-title error">Error</h3>
          <pre class="data-block error">{{ entry.error }}</pre>
        </section>
        
        <!-- System prompt section -->
        <section v-if="entry.system_prompt" class="detail-section">
          <h3 class="section-title">System Prompt</h3>
          <pre class="data-block prompt">{{ entry.system_prompt }}</pre>
        </section>
        
        <!-- Input data section -->
        <section v-if="entry.input_data !== null" class="detail-section">
          <h3 class="section-title">Input Data</h3>
          <pre 
            class="data-block"
            :class="{ 'string-data': isStringData(entry.input_data) }"
          >{{ isStringData(entry.input_data) ? entry.input_data : formatJson(entry.input_data) }}</pre>
        </section>
        
        <!-- Output data section -->
        <section v-if="entry.output_data !== null" class="detail-section">
          <h3 class="section-title">Output Data</h3>
          <pre 
            class="data-block"
            :class="{ 'string-data': isStringData(entry.output_data) }"
          >{{ isStringData(entry.output_data) ? entry.output_data : formatJson(entry.output_data) }}</pre>
        </section>
        
        <!-- Trace info section -->
        <section v-if="entry.trace_id || entry.span_id" class="detail-section">
          <h3 class="section-title">Trace Info</h3>
          <div class="metadata-grid">
            <div v-if="entry.trace_id" class="metadata-item full-width">
              <span class="metadata-label">Trace ID</span>
              <span class="metadata-value mono small">{{ entry.trace_id }}</span>
            </div>
            <div v-if="entry.span_id" class="metadata-item full-width">
              <span class="metadata-label">Span ID</span>
              <span class="metadata-value mono small">{{ entry.span_id }}</span>
            </div>
          </div>
        </section>
        
        <!-- Metadata section -->
        <section v-if="entry.metadata" class="detail-section">
          <h3 class="section-title">Additional Metadata</h3>
          <pre class="data-block">{{ formatJson(entry.metadata) }}</pre>
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.detail-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
  border-left: 1px solid var(--border-subtle);
}

.detail-panel__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 12px;
  color: var(--text-muted);
}

.empty-icon {
  font-size: 36px;
  opacity: 0.5;
}

.empty-text {
  font-size: 13px;
}

.detail-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.header__badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: var(--radius-md);
  color: var(--bg-primary);
  font-weight: 600;
}

.header__icon {
  font-size: 14px;
}

.header__type {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.header__error-badge {
  padding: 4px 10px;
  background: var(--accent-red);
  color: white;
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.detail-panel__content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.detail-section {
  margin-bottom: 24px;
}

.detail-section:last-child {
  margin-bottom: 0;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
}

.section-title.error {
  color: var(--accent-red);
  border-color: var(--accent-red);
}

.metadata-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.metadata-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.metadata-item.full-width {
  grid-column: 1 / -1;
}

.metadata-label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.metadata-value {
  font-size: 13px;
  color: var(--text-primary);
  word-break: break-all;
}

.metadata-value.mono {
  font-family: var(--font-mono);
}

.metadata-value.small {
  font-size: 11px;
}

.metadata-value.highlight {
  color: var(--accent-cyan);
  font-weight: 500;
}

.data-block {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  overflow-x: auto;
  max-height: 300px;
  overflow-y: auto;
}

.data-block.string-data {
  white-space: pre-wrap;
}

.data-block.error {
  background: rgba(255, 85, 85, 0.1);
  border-color: var(--accent-red);
  color: var(--accent-red);
}

.data-block.prompt {
  background: rgba(189, 147, 249, 0.1);
  border-color: var(--accent-purple);
}

.error-section {
  background: rgba(255, 85, 85, 0.05);
  padding: 16px;
  border-radius: var(--radius-md);
  margin-left: -16px;
  margin-right: -16px;
  padding-left: 16px;
  padding-right: 16px;
}
</style>
