<script setup lang="ts">
import { computed } from 'vue'
import { Handle, Position } from '@vue-flow/core'
import { getEventColor, getEventIcon, formatDuration } from '../types/protocol'
import type { EventType } from '../types/protocol'

interface Props {
  id: string
  data: {
    label: string
    event_type: EventType
    entry_id: string
    timestamp: string
    duration_ms: number | null
    tokens: number | null
    has_error: boolean
    selected: boolean
  }
}

const props = defineProps<Props>()

const nodeClass = computed(() => {
  const classes = ['protocol-node']
  
  const eventType = props.data.event_type
  if (eventType.startsWith('agent')) classes.push('agent')
  else if (eventType.startsWith('llm')) classes.push('llm')
  else if (eventType.startsWith('tool')) classes.push('tool')
  
  if (props.data.has_error) classes.push('error')
  if (props.data.selected) classes.push('selected')
  
  return classes.join(' ')
})

const accentColor = computed(() => getEventColor(props.data.event_type))
const icon = computed(() => getEventIcon(props.data.event_type))
</script>

<template>
  <div :class="nodeClass" :style="{ '--accent': accentColor }">
    <Handle type="target" :position="Position.Top" />
    
    <div class="protocol-node__icon">{{ icon }}</div>
    
    <div class="protocol-node__content">
      <div class="protocol-node__label">{{ data.label }}</div>
      <div class="protocol-node__meta">
        <span v-if="data.duration_ms" class="meta-item">
          {{ formatDuration(data.duration_ms) }}
        </span>
        <span v-if="data.tokens" class="meta-item">
          {{ data.tokens }} tok
        </span>
      </div>
    </div>
    
    <Handle type="source" :position="Position.Bottom" />
  </div>
</template>

<style scoped>
.protocol-node {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg-elevated);
  border: 2px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  min-width: 160px;
  box-shadow: var(--shadow-subtle);
  transition: all 0.2s ease;
  cursor: pointer;
}

.protocol-node:hover {
  border-color: var(--accent);
  box-shadow: 0 0 12px color-mix(in srgb, var(--accent) 30%, transparent);
}

.protocol-node.selected {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
}

.protocol-node.agent {
  border-left: 4px solid var(--node-agent);
}

.protocol-node.llm {
  border-left: 4px solid var(--node-llm);
}

.protocol-node.tool {
  border-left: 4px solid var(--node-tool);
}

.protocol-node.error {
  border-color: var(--node-error);
  background: rgba(255, 85, 85, 0.1);
}

.protocol-node__icon {
  font-size: 18px;
  flex-shrink: 0;
}

.protocol-node__content {
  flex: 1;
  min-width: 0;
}

.protocol-node__label {
  font-weight: 500;
  font-size: 13px;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.protocol-node__meta {
  display: flex;
  gap: 8px;
  margin-top: 2px;
}

.meta-item {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
}

:deep(.vue-flow__handle) {
  width: 8px;
  height: 8px;
  background: var(--border-color);
  border: 2px solid var(--bg-elevated);
}

:deep(.vue-flow__handle:hover) {
  background: var(--accent-cyan);
}
</style>
