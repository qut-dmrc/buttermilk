# DaisyUI Implementation for Buttermilk Dashboard

## Overview

This dashboard has been upgraded to use DaisyUI components, a plugin for Tailwind CSS. DaisyUI provides a comprehensive set of accessible, reusable components that are easy to customize.

## Components Used

The following DaisyUI components have been implemented:

- **Card**: For all major container elements
- **Button**: Various button styles with states
- **Form Controls**: Input fields, selects, and labels
- **Badge**: For status indicators and labels
- **Collapse**: For expandable/collapsible sections
- **Tooltip**: For displaying additional information on hover
- **Modal**: For detailed views of scores and predictions
- **Alert**: For system messages and notifications
- **Loading**: For loading indicators
- **Join**: For input groups like search/submit combinations

## Theme

The dashboard uses the default DaisyUI light theme, with the following color classes:

- `bg-base-100`: Primary background
- `bg-base-200`: Secondary/container background
- `text-base-content`: Main text color
- `border-base-300`: Border color

## File Structure

- `base.html`: Contains the DaisyUI setup and theme configuration
- `index.html`: Main dashboard layout and controls
- `partials/outcomes_panel.html`: Displays predictions and scores
- Various JavaScript files for handling interactions

## CSS Classes

DaisyUI classes follow a consistent pattern:

```
btn                  # Component base
btn-primary          # Component variant
btn-sm               # Component size
```

For example, to create a small success button:

```html
<button class="btn btn-success btn-sm">Success</button>
```

## Responsive Design

All components adapt to different screen sizes. The main layout uses a column-based approach on mobile and switches to a row-based layout on larger screens.
