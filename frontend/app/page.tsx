'use client';

import { useState, useRef, useEffect } from 'react';
import { PanelGroup, Panel, PanelResizeHandle, ImperativePanelHandle } from 'react-resizable-panels';
import { NodeGraphSimple } from '@/components/NodeGraph/NodeGraphSimple';
import { SystemControls } from '@/components/Controls/SystemControls';
import { Tooltip } from '@/components/UI/Tooltip';

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const sidebarRef = useRef<ImperativePanelHandle>(null);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024); // Phones & tablets (< 1024px) use mobile overlay
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const toggleSidebar = () => {
    if (isMobile) {
      // On mobile, directly toggle the state
      setSidebarCollapsed(!sidebarCollapsed);
    } else {
      // On desktop, use the panel's collapse/expand methods
      const panel = sidebarRef.current;
      if (panel) {
        if (sidebarCollapsed) {
          panel.expand();
        } else {
          panel.collapse();
        }
      }
    }
  };

  return (
    <div className="flex flex-col h-screen text-[#00ff99] font-mono overflow-hidden">
      {/* Top Bar */}
      <div className="flex items-center justify-between px-2 sm:px-4 py-2 sm:py-3 border-b border-[#00cc77] bg-black/80 backdrop-blur-md flex-shrink-0 shadow-glow z-10 gap-2">
        <div className="flex items-center gap-1.5 sm:gap-2 md:gap-3 overflow-hidden min-w-0 flex-1">
          <div className="text-[10px] sm:text-xs md:text-sm font-semibold text-[#00ff99] glow-text truncate">
            GMCS - Chaotic-Neuro Computational Substrate
            <span className="hidden md:inline"> Platform</span>
          </div>
          <div className="text-[10px] sm:text-xs text-[#00ff99] bg-black px-1 sm:px-1.5 md:px-2 py-0.5 sm:py-1 rounded border border-[#00cc77] flex-shrink-0">
            v0.1
          </div>
          <div className="text-xs text-[#00cc77] hidden lg:block flex-shrink-0">
            In Development
          </div>
        </div>
        <div className="flex items-center gap-1 sm:gap-1.5 md:gap-2 text-xs text-[#00ff99] flex-shrink-0">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-[#00ff99] animate-pulse shadow-glow-md" />
            <span className="hidden md:inline text-[10px] sm:text-xs">CONNECTED</span>
          </div>
          <div className="border-l border-[#00cc77] h-4 hidden md:block" />
          <span className="hidden md:inline text-[10px] sm:text-xs">T = 0.00s</span>
          <div className="border-l border-[#00cc77] h-4 hidden lg:block" />
          <span className="hidden lg:inline text-[10px] sm:text-xs">FPS = 60</span>
          <Tooltip content={sidebarCollapsed ? "Show Controls" : "Hide Controls"} position="bottom">
            <button
              onClick={toggleSidebar}
              className="ml-1 px-1.5 sm:px-2 py-1 text-[10px] sm:text-xs bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold flex-shrink-0"
            >
              {sidebarCollapsed ? '◀' : '▶'}
            </button>
          </Tooltip>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 relative overflow-hidden">
        {/* Desktop: Resizable Panels */}
        {!isMobile ? (
          <PanelGroup direction="horizontal" className="h-full">
            {/* Main Node Graph Area */}
            <Panel
              defaultSize={78}
              minSize={50}
              className="flex flex-col"
            >
              <NodeGraphSimple />
            </Panel>

            <PanelResizeHandle className="resize-handle-vertical" />

            {/* Right Sidebar - System Controls */}
            <Panel
              ref={sidebarRef}
              defaultSize={22}
              minSize={15}
              maxSize={50}
              className="flex flex-col"
              collapsible={true}
              collapsedSize={0}
              onCollapse={() => setSidebarCollapsed(true)}
              onExpand={() => setSidebarCollapsed(false)}
            >
              {!sidebarCollapsed && (
                <div className="h-full border-l border-[#00cc77] bg-black/80 backdrop-blur-md shadow-glow overflow-auto custom-scrollbar">
                  <SystemControls />
                </div>
              )}
            </Panel>
          </PanelGroup>
        ) : (
          /* Mobile: Full-width node graph with overlay sidebar */
          <>
            <NodeGraphSimple />

            {/* Mobile Sidebar Overlay */}
            {!sidebarCollapsed && (
              <>
                {/* Backdrop */}
                <div
                  className="absolute inset-0 bg-black/50 backdrop-blur-sm z-40"
                  onClick={() => setSidebarCollapsed(true)}
                />

                {/* Sidebar */}
                <div className="absolute top-0 right-0 bottom-0 w-[70%] md:w-[50%] border-l border-[#00cc77] bg-black/95 backdrop-blur-md shadow-glow overflow-auto custom-scrollbar z-50 animate-slide-in">
                  <SystemControls />
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
