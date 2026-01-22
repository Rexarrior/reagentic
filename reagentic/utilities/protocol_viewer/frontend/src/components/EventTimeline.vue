<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue'
import { format, parseISO, differenceInMilliseconds } from 'date-fns'
import { useProtocolStore } from '../stores/protocol'
import { getEventColor, getEventIcon, formatDuration } from '../types/protocol'
import type { ProtocolEntry, EventType } from '../types/protocol'

const store = useProtocolStore()
const timelineRef = ref<HTMLElement | null>(null)

const entries = computed(() => store.entries)

// Calculate positions for timeline items
const timelineData = computed(() => {
  if (!entries.value.length) return []
  
  const sorted = [...entries.value].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )
  
  const startTime = new Date(sorted[0].timestamp).getTime()
  const endTime = new Date(sorted[sorted.length - 1].timestamp).getTime()
  const totalDuration = endTime - startTime || 1 // Avoid division by zero
  
  return sorted.map((entry, index) => {
    const entryTime = new Date(entry.timestamp).getTime()
    const position = ((entryTime - startTime) / totalDuration) * 100
    
    return {
      entry,
      position: Math.max(2, Math.min(98, position)), // Keep within bounds
      index,
    }
  })
})

// Total duration of the session
const totalDuration = computed(() => {
  if (!entries.value.length) return null
  const sorted = [...entries.value].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  )
  return differenceInMilliseconds(
    parseISO(sorted[sorted.length - 1].timestamp),
    parseISO(sorted[0].timestamp)
  )
})

function selectEntry(entry: ProtocolEntry) {
  store.selectEntry(entry.id)
  scrollToSelected()
}

function scrollToSelected() {
  nextTick(() => {
    const selectedEl = timelineRef.value?.querySelector('.timeline-item--selected')
    if (selectedEl) {
      selectedEl.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
    }
  })
}

function formatTimestamp(isoString: string): string {
  return format(parseISO(isoString), 'HH:mm:ss.SSS')
}

function isSelected(entry: ProtocolEntry): boolean {
  return store.selectedEntryId === entry.id
}

// Keyboard navigation
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'ArrowLeft') {
    store.selectPrevEntry()
    scrollToSelected()
  } else if (event.key === 'ArrowRight') {
    store.selectNextEntry()
    scrollToSelected()
  }
}

// Watch for selection changes to scroll
watch(() => store.selectedEntryId, scrollToSelected)
</script>

<template>
  <div 
    class="event-timeline" 
    ref="timelineRef"
    tabindex="0"
    @keydown="handleKeydown"
  >
    <div v-if="!store.hasSession" class="timeline__empty">
      Select a session to view timeline
    </div>
    
    <template v-else>
      <div class="timeline__header">
        <div class="timeline__title">
          <span class="timeline__label">Timeline</span>
          <span v-if="totalDuration !== null" class="timeline__duration">
            {{ formatDuration(totalDuration) }}
          </span>
        </div>
        
        <div class="timeline__nav">
          <button 
            class="nav-btn" 
            @click="store.selectPrevEntry"
            :disabled="!store.selectedEntryId"
          >
            ←
          </button>
          <span class="nav-counter">
            {{ timelineData.findIndex(t => t.entry.id === store.selectedEntryId) + 1 }}
            / {{ timelineData.length }}
          </span>
          <button 
            class="nav-btn" 
            @click="store.selectNextEntry"
            :disabled="!store.selectedEntryId"
          >
            →
          </button>
        </div>
      </div>
      
      <div class="timeline__track">
        <div class="timeline__line"></div>
        
        <div
          v-for="{ entry, position } in timelineData"
          :key="entry.id"
          class="timeline-item"
          :class="{
            'timeline-item--selected': isSelected(entry),
            'timeline-item--error': entry.error
          }"
          :style="{ left: `${position}%` }"
          @click="selectEntry(entry)"
        >
          <div 
            class="timeline-item__dot"
            :style="{ backgroundColor: getEventColor(entry.event_type as EventType) }"
          >
            <span class="timeline-item__icon">{{ getEventIcon(entry.event_type as EventType) }}</span>
          </div>
          
          <div class="timeline-item__tooltip">
            <div class="tooltip__type">{{ entry.event_type }}</div>
            <div class="tooltip__time">{{ formatTimestamp(entry.timestamp) }}</div>
            <div v-if="entry.duration_ms" class="tooltip__duration">
              {{ formatDuration(entry.duration_ms) }}
            </div>
          </div>
        </div>
      </div>
      
      <div class="timeline__items">
        <div
          v-for="{ entry } in timelineData"
          :key="entry.id"
          class="timeline-list-item"
          :class="{ 'timeline-list-item--selected': isSelected(entry) }"
          @click="selectEntry(entry)"
        >
          <span 
            class="timeline-list-item__icon"
            :style="{ color: getEventColor(entry.event_type as EventType) }"
          >
            {{ getEventIcon(entry.event_type as EventType) }}
          </span>
          <span class="timeline-list-item__type">{{ entry.event_type }}</span>
          <span class="timeline-list-item__name">
            {{ entry.agent_name || entry.tool_name || '' }}
          </span>
          <span class="timeline-list-item__time">
            {{ formatTimestamp(entry.timestamp) }}
          </span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.event-timeline {
  display: flex;
  flex-direction: column;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-subtle);
  height: 100%;
  outline: none;
}

.event-timeline:focus {
  box-shadow: inset 0 0 0 2px var(--accent-cyan);
}

.timeline__empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted);
  font-size: 13px;
}

.timeline__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-subtle);
}

.timeline__title {
  display: flex;
  align-items: center;
  gap: 12px;
}

.timeline__label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-secondary);
}

.timeline__duration {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  padding: 2px 8px;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
}

.timeline__nav {
  display: flex;
  align-items: center;
  gap: 8px;
}

.nav-btn {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.nav-btn:hover:not(:disabled) {
  background: var(--bg-elevated);
  color: var(--text-primary);
  border-color: var(--accent-cyan);
}

.nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.nav-counter {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-muted);
  min-width: 60px;
  text-align: center;
}

.timeline__track {
  position: relative;
  height: 60px;
  padding: 20px 40px;
  flex-shrink: 0;
}

.timeline__line {
  position: absolute;
  left: 40px;
  right: 40px;
  top: 50%;
  height: 3px;
  background: var(--border-color);
  border-radius: 2px;
}

.timeline-item {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  z-index: 1;
  cursor: pointer;
}

.timeline-item__dot {
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 3px solid var(--bg-secondary);
  transition: all 0.2s;
}

.timeline-item__icon {
  font-size: 10px;
  opacity: 0;
  transition: opacity 0.2s;
}

.timeline-item:hover .timeline-item__dot,
.timeline-item--selected .timeline-item__dot {
  transform: scale(1.3);
  border-width: 2px;
}

.timeline-item:hover .timeline-item__icon,
.timeline-item--selected .timeline-item__icon {
  opacity: 1;
}

.timeline-item--selected .timeline-item__dot {
  box-shadow: 0 0 0 4px rgba(0, 217, 255, 0.3);
}

.timeline-item--error .timeline-item__dot {
  background: var(--accent-red) !important;
}

.timeline-item__tooltip {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  margin-bottom: 8px;
  padding: 8px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s;
  pointer-events: none;
  z-index: 10;
}

.timeline-item:hover .timeline-item__tooltip {
  opacity: 1;
  visibility: visible;
}

.tooltip__type {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
}

.tooltip__time {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}

.tooltip__duration {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent-green);
  margin-top: 2px;
}

.timeline__items {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.timeline-list-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s;
}

.timeline-list-item:hover {
  background: var(--bg-tertiary);
}

.timeline-list-item--selected {
  background: var(--bg-elevated);
  border-left: 3px solid var(--accent-cyan);
}

.timeline-list-item__icon {
  font-size: 14px;
  width: 20px;
  text-align: center;
}

.timeline-list-item__type {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
  min-width: 100px;
}

.timeline-list-item__name {
  font-size: 12px;
  color: var(--text-secondary);
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.timeline-list-item__time {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-muted);
}
</style>
