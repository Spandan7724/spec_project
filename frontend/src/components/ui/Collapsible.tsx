import { useState, ReactNode } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface CollapsibleProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
  onMobileOnly?: boolean; // Only collapsible on mobile
}

export default function Collapsible({
  title,
  children,
  defaultOpen = false,
  className = '',
  onMobileOnly = false,
}: CollapsibleProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const toggleOpen = () => {
    if (onMobileOnly) {
      // On desktop (md and above), always show content
      const isDesktop = window.innerWidth >= 768;
      if (isDesktop) return;
    }
    setIsOpen(!isOpen);
  };

  // For mobile-only collapsibles, always show on desktop
  const shouldShowContent = onMobileOnly
    ? (typeof window !== 'undefined' && window.innerWidth >= 768) || isOpen
    : isOpen;

  return (
    <div className={`border rounded-lg ${className}`}>
      <button
        onClick={toggleOpen}
        className={`mobile-tap w-full flex items-center justify-between p-4 text-left hover:bg-muted/50 transition-colors ${
          onMobileOnly ? 'md:cursor-default md:hover:bg-transparent' : ''
        }`}
        aria-expanded={shouldShowContent}
      >
        <h3 className="font-semibold text-sm md:text-base">{title}</h3>
        <span className={onMobileOnly ? 'md:hidden' : ''}>
          {isOpen ? (
            <ChevronUp className="h-5 w-5 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-5 w-5 text-muted-foreground" />
          )}
        </span>
      </button>

      <div
        className={`overflow-hidden transition-all duration-300 ease-in-out ${
          shouldShowContent ? 'max-h-[9999px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="p-4 pt-0 border-t">
          {children}
        </div>
      </div>
    </div>
  );
}

// Simpler variant with just show more/less button
interface ShowMoreProps {
  children: ReactNode;
  previewHeight?: string; // e.g., "200px"
  buttonText?: string;
  className?: string;
}

export function ShowMore({
  children,
  previewHeight = '200px',
  buttonText = 'Show more',
  className = '',
}: ShowMoreProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={className}>
      <div
        className={`overflow-hidden transition-all duration-300 ${
          isExpanded ? 'max-h-[9999px]' : ''
        }`}
        style={!isExpanded ? { maxHeight: previewHeight } : {}}
      >
        {children}
      </div>

      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="mobile-tap mt-2 text-sm font-medium text-primary hover:underline"
      >
        {isExpanded ? 'Show less' : buttonText}
      </button>
    </div>
  );
}
