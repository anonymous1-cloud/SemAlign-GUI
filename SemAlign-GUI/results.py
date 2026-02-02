#!/usr/bin/env python3
"""
GUI Change Analysis - Single Comprehensive Figure for PPT
Optimized layout with all text boxes within boundaries
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import matplotlib
import re
import time
import warnings
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore')

# Set very large fonts for PPT presentation
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Tahoma']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 18  # Very large base font size
matplotlib.rcParams['axes.titlesize'] = 22  # Title font size
matplotlib.rcParams['axes.labelsize'] = 20  # Axis label font size
matplotlib.rcParams['xtick.labelsize'] = 16  # X tick font size
matplotlib.rcParams['ytick.labelsize'] = 16  # Y tick font size
matplotlib.rcParams['legend.fontsize'] = 18  # Legend font size
matplotlib.rcParams['figure.titlesize'] = 30  # Figure title font size


class PPTGUIAnalyzer:
    """GUI Analysis for PPT - Single Comprehensive Figure with large fonts"""

    def __init__(self):
        # Color mapping
        self.colors = {
            'addition': '#4CAF50',  # Green - Added
            'removal': '#F44336',  # Red - Removed
            'movement': '#2196F3',  # Blue - Moved
            'TextView': '#FF9800',  # Orange
            'ImageView': '#3F51B5',  # Dark blue
            'Button': '#009688',  # Teal
            'EditText': '#9C27B0',  # Purple
            'WebView': '#795548',  # Brown
            'View': '#607D8B',  # Blue gray
            'success': '#4CAF50',  # Success - Green
            'warning': '#FF9800',  # Warning - Orange
            'error': '#F44336',  # Error - Red
            'heatmap_high': '#FF0000',  # Red - High intensity
            'heatmap_medium': '#FF9900',  # Orange - Medium intensity
            'heatmap_low': '#FFFF00',  # Yellow - Low intensity
        }

        # Component type icons mapping
        self.component_icons = {
            'TextView': '📝',
            'ImageView': '🖼️',
            'Button': '🔄',
            'EditText': '📝',
            'WebView': '🌐',
            'View': '▢'
        }

    def get_component_icon(self, component_type: str) -> str:
        """Get component icon"""
        return self.component_icons.get(component_type, '▢')

    def get_component_color(self, component_type: str, change_type: str) -> str:
        """Get component color"""
        if change_type in self.colors:
            return self.colors[change_type]
        elif component_type in self.colors:
            return self.colors[component_type]
        else:
            return '#757575'

    def create_comprehensive_figure(self, results: Dict, output_path: str = "gui_analysis_ppt.png"):
        """Create single comprehensive figure for PPT presentation with optimized layout"""
        print("\nCreating comprehensive figure for PPT...")

        # Create figure with optimized size for PPT (wider format)
        fig = plt.figure(figsize=(28, 24))  # Increased size to accommodate more space

        # Use GridSpec: 3 main rows, 3 columns - adjust height ratios
        gs = gridspec.GridSpec(
            3, 3,  # 3 rows, 3 columns
            height_ratios=[1.5, 1.5, 1.2],  # Adjusted to give more space to first two rows
            width_ratios=[1, 1, 1],  # Equal columns
            hspace=0.55,  # Even more vertical spacing
            wspace=0.4  # Significantly increased horizontal spacing
        )

        # Extract data
        if Path(results['reference_image']).exists():
            ref_img = Image.open(results['reference_image']).convert('RGB')
        else:
            ref_img = self.create_demo_image()

        if Path(results['target_image']).exists():
            tar_img = Image.open(results['target_image']).convert('RGB')
        else:
            tar_img = self.create_demo_image()

        changes = results['detailed_changes']
        visual = results['visual_analysis']
        overall = results['overall_assessment']

        # Resize images for display
        display_size = (340, 340)  # Increased size for better visibility
        ref_img_disp = ref_img.resize(display_size)
        tar_img_disp = tar_img.resize(display_size)

        # ========== Row 1: Reference image + Heatmap + Text block 1 ==========
        print("  Row 1: Reference image, heatmap, and summary...")

        # 1.1 Reference image (left)
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(ref_img_disp)
        ax1.set_title('Reference Image\n(Original GUI)', fontsize=24, fontweight='bold', pad=25)
        ax1.axis('off')

        # Annotate removed components with careful positioning
        removed_count = 0
        for removal in changes['removals']:
            if removed_count >= 2:  # Show up to 2 only
                break
            removed_count += 1

            bbox = removal['bbox']
            scale_x = display_size[0] / ref_img.size[0]
            scale_y = display_size[1] / ref_img.size[1]
            scaled_bbox = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]

            # Draw bounding box
            rect = patches.Rectangle(
                (scaled_bbox[0], scaled_bbox[1]),
                scaled_bbox[2] - scaled_bbox[0],
                scaled_bbox[3] - scaled_bbox[1],
                linewidth=3, edgecolor=self.colors['removal'],
                facecolor='none', linestyle='--', alpha=0.8
            )
            ax1.add_patch(rect)

            # Add label - position carefully to avoid overlap
            icon = self.get_component_icon(removal['type'])
            label = f"{icon} {removal['type']}"

            # Calculate text position - place above or below based on position
            text_x = max(10, min(scaled_bbox[0] + 10, display_size[0] - 100))

            # Place above if near bottom, otherwise below
            if scaled_bbox[1] > display_size[1] * 0.7:
                text_y = scaled_bbox[1] - 20
            else:
                text_y = scaled_bbox[1] + 20

            ax1.text(
                text_x, text_y, label,
                color='white', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=self.colors['removal'],
                          alpha=0.9, linewidth=2, edgecolor='white')
            )

        # Add removal count at top
        if changes['removals']:
            ax1.text(0.5, 0.98, f"Removed: {len(changes['removals'])}",
                     transform=ax1.transAxes, ha='center', color='white', fontsize=18, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['removal'],
                               alpha=0.9, linewidth=2, edgecolor='white'))

        # 1.2 Heatmap (middle)
        ax2 = plt.subplot(gs[0, 1])

        # Get heatmap data
        diff_map = visual.get('diff_map', None)

        if diff_map is not None:
            # Create heatmap with clear spacing
            im = ax2.imshow(diff_map, cmap='hot', aspect='auto', vmin=0, vmax=1)

            # Set titles and labels with padding
            ax2.set_title('Change Intensity Heatmap', fontsize=24, fontweight='bold', pad=25)
            ax2.set_xlabel('X Coordinate', fontsize=20, labelpad=15)
            ax2.set_ylabel('Y Coordinate', fontsize=20, labelpad=15)

            # Add grid for better readability
            ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

            # Add colorbar with padding
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.08)
            cbar.set_label('Change Intensity', fontsize=18, labelpad=15)
            cbar.ax.tick_params(labelsize=16)

            # Add key statistics in top-left corner with padding
            stats_text = f"Change area: {visual.get('change_area_percentage', 0):.2%}\n"
            stats_text += f"Max intensity: {visual.get('change_intensity_stats', {}).get('max_intensity', 0):.3f}"

            ax2.text(0.05, 0.97, stats_text, fontsize=17,
                     transform=ax2.transAxes, color='white',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7),
                     verticalalignment='top')
        else:
            ax2.text(0.5, 0.5, 'No heatmap data', ha='center', va='center',
                     fontsize=22, fontweight='bold')
            ax2.set_title('Change Intensity Heatmap', fontsize=24, fontweight='bold', pad=25)

        # 1.3 Text block 1 - Change summary (right) - COMPACTED
        ax3 = plt.subplot(gs[0, 2])
        ax3.axis('off')

        # Build summary text - more compact format
        summary_text = "📊 Analysis Summary\n"
        summary_text += "=" * 18 + "\n\n"

        summary_text += "🔢 Change Statistics:\n"
        summary_text += f"• Total: {changes['statistics']['total_changes']}\n"
        summary_text += f"• Added: {len(changes['additions'])}\n"
        summary_text += f"• Removed: {len(changes['removals'])}\n"
        summary_text += f"• Moved: {len(changes['movements'])}\n\n"

        summary_text += "👁️ Visual Analysis:\n"
        change_prob = visual.get('change_probability', 0)
        summary_text += f"• Probability: {change_prob:.2%}\n"
        summary_text += f"• Change: {'✅ Yes' if visual.get('has_change', False) else '❌ No'}\n"
        summary_text += f"• Area: {visual.get('change_area_percentage', 0):.2%}\n\n"

        summary_text += "📏 Image Sizes:\n"
        summary_text += f"• Ref: {ref_img.size[0]}×{ref_img.size[1]}px\n"
        summary_text += f"• Target: {tar_img.size[0]}×{tar_img.size[1]}px\n\n"

        summary_text += f"🕒 Time: {results.get('analysis_time', 0):.1f}s"

        # Use smaller font and tighter spacing
        ax3.text(0.05, 0.95, summary_text, fontsize=17,
                 verticalalignment='top', transform=ax3.transAxes, linespacing=1.3,
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa',
                           edgecolor='#dee2e6', alpha=0.95, linewidth=3))

        # ========== Row 2: Target image + Mask + Text block 2 ==========
        print("  Row 2: Target image, mask, and coordinates...")

        # 2.1 Target image (left)
        ax4 = plt.subplot(gs[1, 0])
        ax4.imshow(tar_img_disp)
        ax4.set_title('Target Image\n(Modified GUI)', fontsize=24, fontweight='bold', pad=25)
        ax4.axis('off')

        # Annotate added components
        added_count = 0
        for addition in changes['additions']:
            if added_count >= 2:  # Show up to 2 only
                break
            added_count += 1

            bbox = addition['bbox']
            scale_x = display_size[0] / tar_img.size[0]
            scale_y = display_size[1] / tar_img.size[1]
            scaled_bbox = [
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]

            # Draw bounding box
            rect = patches.Rectangle(
                (scaled_bbox[0], scaled_bbox[1]),
                scaled_bbox[2] - scaled_bbox[0],
                scaled_bbox[3] - scaled_bbox[1],
                linewidth=3, edgecolor=self.colors['addition'],
                facecolor='none', linestyle='-', alpha=0.8
            )
            ax4.add_patch(rect)

            # Add label
            icon = self.get_component_icon(addition['type'])
            label = f"{icon} {addition['type']}"

            # Calculate text position
            text_x = max(10, min(scaled_bbox[0] + 10, display_size[0] - 100))

            # Place above if near bottom, otherwise below
            if scaled_bbox[1] > display_size[1] * 0.7:
                text_y = scaled_bbox[1] - 20
            else:
                text_y = scaled_bbox[1] + 20

            ax4.text(
                text_x, text_y, label,
                color='white', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=self.colors['addition'],
                          alpha=0.9, linewidth=2, edgecolor='white')
            )

        # Draw movement trajectories
        movement_count = 0
        for movement in changes['movements']:
            movement_count += 1
            color = self.colors['movement']
            icon = self.get_component_icon(movement['type'])

            # Scale coordinates
            scale_x = display_size[0] / tar_img.size[0]
            scale_y = display_size[1] / tar_img.size[1]

            from_center = movement['from_center']
            to_center = movement['to_center']

            scaled_from = [from_center[0] * scale_x, from_center[1] * scale_y]
            scaled_to = [to_center[0] * scale_x, to_center[1] * scale_y]

            # Draw arrow
            arrow = FancyArrowPatch(
                scaled_from, scaled_to,
                arrowstyle='->', color=color, linewidth=3,
                mutation_scale=25, alpha=0.8
            )
            ax4.add_patch(arrow)

            # Add movement number with careful positioning
            label_x = scaled_to[0] + 10
            label_y = scaled_to[1] + 10

            # Ensure label is within image bounds
            label_x = min(label_x, display_size[0] - 25)
            label_y = min(label_y, display_size[1] - 25)

            ax4.text(
                label_x, label_y, f"M{movement_count}",
                color='white', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.4', facecolor=color,
                          alpha=0.9, linewidth=2, edgecolor='white')
            )

        # Add addition count at top
        if changes['additions']:
            ax4.text(0.5, 0.98, f"Added: {len(changes['additions'])}",
                     transform=ax4.transAxes, ha='center', color='white', fontsize=18, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['addition'],
                               alpha=0.9, linewidth=2, edgecolor='white'))

        # 2.2 Change mask analysis (middle)
        ax5 = plt.subplot(gs[1, 1])

        change_mask = visual.get('change_mask', None)

        if change_mask is not None:
            # Create mask visualization
            im_mask = ax5.imshow(change_mask, cmap='binary', aspect='auto')
            ax5.set_title(f'Change Detection Mask\n(Threshold: {visual.get("threshold_used", 0.1):.2f})',
                          fontsize=24, fontweight='bold', pad=25)
            ax5.set_xlabel('X Coordinate', fontsize=20, labelpad=15)
            ax5.set_ylabel('Y Coordinate', fontsize=20, labelpad=15)

            # Add grid
            ax5.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

            # Add mask statistics with padding
            changed_pixels = int(change_mask.sum())
            total_pixels = change_mask.size

            mask_stats = f"Changed pixels: {changed_pixels:,}\n"
            mask_stats += f"Total pixels: {total_pixels:,}\n"
            mask_stats += f"Change ratio: {change_mask.mean():.2%}\n"
            mask_stats += f"Regions: {len(visual.get('change_regions', []))}"

            # Place text in bottom-left with padding
            ax5.text(0.05, 0.05, mask_stats, fontsize=17,
                     transform=ax5.transAxes, color='white',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7),
                     verticalalignment='bottom')

            # Mark key change regions with labels above
            regions = visual.get('change_regions', [])
            for region in regions[:2]:  # Mark top 2 regions
                bbox = region['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                    linewidth=2, edgecolor='red', facecolor='none', alpha=0.7
                )
                ax5.add_patch(rect)

                # Add region label above the region
                ax5.text(bbox[0], bbox[1] - 25, f"Region {region['id']}",
                         color='red', fontsize=15, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No mask data', ha='center', va='center',
                     fontsize=22, fontweight='bold')
            ax5.set_title('Change Detection Mask', fontsize=24, fontweight='bold', pad=25)

        # 2.3 Text block 2 - Component coordinates (right) - FURTHER COMPACTED
        ax6 = plt.subplot(gs[1, 2])
        ax6.axis('off')

        coordinates_text = "📍 Component Coordinates\n"
        coordinates_text += "=" * 22 + "\n\n"

        # Show very limited number for clarity
        max_components = 2  # Reduced from 3 to 2

        # Additions with coordinates - super compact format
        if changes['additions']:
            coordinates_text += "✅ Additions:\n"
            for i, addition in enumerate(changes['additions'][:max_components]):
                icon = self.get_component_icon(addition['type'])
                bbox = addition['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # Super compact format - single line
                coordinates_text += f"{icon} {addition['type']}: "
                coordinates_text += f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})\n"
                coordinates_text += f"  {width}×{height}px\n"

        # Removals with coordinates - super compact format
        if changes['removals']:
            coordinates_text += "\n❌ Removals:\n"
            for i, removal in enumerate(changes['removals'][:max_components]):
                icon = self.get_component_icon(removal['type'])
                bbox = removal['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                # Super compact format - single line
                coordinates_text += f"{icon} {removal['type']}: "
                coordinates_text += f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})\n"
                coordinates_text += f"  {width}×{height}px\n"

        # Add note if there are more components - on same line
        total_added = len(changes['additions'])
        total_removed = len(changes['removals'])

        extra_info = []
        if total_added > max_components:
            extra_info.append(f"+{total_added - max_components} more adds")
        if total_removed > max_components:
            extra_info.append(f"+{total_removed - max_components} more removes")

        if extra_info:
            coordinates_text += f"\n{' | '.join(extra_info)}"

        # Use even smaller font and tighter spacing
        ax6.text(0.05, 0.95, coordinates_text, fontsize=16,  # Reduced from 17 to 16
                 verticalalignment='top', transform=ax6.transAxes, linespacing=1.2,  # Reduced line spacing
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8',  # Reduced padding
                           edgecolor='#b6d4e3', alpha=0.95, linewidth=3))

        # ========== Row 3: Remaining text blocks ==========
        print("  Row 3: Remaining text blocks...")

        # 3.1 Movement details (left)
        ax7 = plt.subplot(gs[2, 0])
        ax7.axis('off')

        if changes['movements']:
            movements_text = "🔄 Movement Details\n"
            movements_text += "=" * 20 + "\n\n"

            for i, movement in enumerate(changes['movements']):
                icon = self.get_component_icon(movement['type'])

                from_bbox = movement['from_bbox']
                to_bbox = movement['to_bbox']

                # Calculate movement distance
                from_center = movement['from_center']
                to_center = movement['to_center']
                distance = ((to_center[0] - from_center[0]) ** 2 +
                            (to_center[1] - from_center[1]) ** 2) ** 0.5

                movements_text += f"M{i + 1}: {icon} {movement['type']}\n"
                movements_text += f"  From: ({from_bbox[0]},{from_bbox[1]})-({from_bbox[2]},{from_bbox[3]})\n"
                movements_text += f"  To: ({to_bbox[0]},{to_bbox[1]})-({to_bbox[2]},{to_bbox[3]})\n"
                movements_text += f"  Dist: {distance:.1f} px\n\n"

                # Limit to 1 movement to avoid overflow
                if i >= 0 and len(changes['movements']) > 1:
                    movements_text += f"... +{len(changes['movements']) - 1} more movements"
                    break
        else:
            movements_text = "🔄 Movement Details\n"
            movements_text += "=" * 20 + "\n\n"
            movements_text += "No components were moved"

        ax7.text(0.05, 0.95, movements_text, fontsize=17,
                 verticalalignment='top', transform=ax7.transAxes, linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.7', facecolor='#e3f2fd',  # Reduced padding
                           edgecolor='#bbdefb', alpha=0.95, linewidth=3))

        # 3.2 Validation results (middle)
        ax8 = plt.subplot(gs[2, 1])
        ax8.axis('off')

        validation_text = "✅ Validation Results\n"
        validation_text += "=" * 20 + "\n\n"

        # Alignment score
        alignment_score = overall.get('alignment_score', 0)

        # Overall confidence
        overall_confidence = overall.get('overall_confidence', 0)

        # Validation status
        change_validated = overall.get('change_validated', False)
        validation_status = "✅ PASS" if change_validated else "❌ FAIL"
        validation_color = self.colors['success'] if change_validated else self.colors['error']

        # Compact progress bars
        def progress_bar(score, width=10):
            filled = int(score * width)
            bar = "█" * filled + "░" * (width - filled)
            return f"{bar} {score:.1%}"

        validation_text += f"Alignment:\n{progress_bar(alignment_score)}\n\n"
        validation_text += f"Confidence:\n{progress_bar(overall_confidence)}\n\n"
        validation_text += f"Status: {validation_status}\n\n"

        validation_text += "📋 Key Findings:\n"
        if change_validated and alignment_score > 0.8:
            validation_text += "• High confidence\n"
            validation_text += "• Changes verified\n"
            validation_text += "• Ready to implement"
        elif change_validated and alignment_score > 0.5:
            validation_text += "• Moderate match\n"
            validation_text += "• Minor issues\n"
            validation_text += "• Review needed"
        elif not change_validated and overall['described_changes_count'] > 0:
            validation_text += "• Validation failed\n"
            validation_text += "• Mismatch found\n"
            validation_text += "• Review required"
        else:
            validation_text += "• No changes\n"
            validation_text += "• Check inputs"

        ax8.text(0.05, 0.95, validation_text, fontsize=17,
                 verticalalignment='top', transform=ax8.transAxes, linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.7', facecolor=validation_color,  # Reduced padding
                           alpha=0.15, edgecolor=validation_color, linewidth=3))

        # 3.3 Intensity distribution and insights (right)
        ax9 = plt.subplot(gs[2, 2])
        ax9.axis('off')

        insights_text = "📈 Change Analysis\n"
        insights_text += "=" * 18 + "\n\n"

        # Heatmap insights
        intensity_stats = visual.get('change_intensity_stats', {})
        if intensity_stats:
            insights_text += "Intensity:\n"
            high_pct = intensity_stats.get('high', 0) * 100
            med_pct = intensity_stats.get('medium', 0) * 100
            low_pct = intensity_stats.get('low', 0) * 100

            insights_text += f"  High: {high_pct:4.1f}%\n"
            insights_text += f"  Med: {med_pct:4.1f}%\n"
            insights_text += f"  Low: {low_pct:4.1f}%\n\n"

            insights_text += f"Max: {intensity_stats.get('max_intensity', 0):.3f}\n"
            insights_text += f"Mean: {intensity_stats.get('mean_intensity', 0):.3f}\n\n"

        # Component change insights
        insights_text += "Components:\n"
        insights_text += f"• Changed: {len(changes['all_components'])}\n"
        insights_text += f"• Added: {len(changes['additions'])}\n"
        insights_text += f"• Removed: {len(changes['removals'])}\n"
        insights_text += f"• Moved: {len(changes['movements'])}\n\n"

        # Analysis metadata
        insights_text += f"Time: {time.strftime('%Y-%m-%d %H:%M')}"

        ax9.text(0.05, 0.95, insights_text, fontsize=17,
                 verticalalignment='top', transform=ax9.transAxes, linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.7', facecolor='#f5f5f5',  # Reduced padding
                           edgecolor='#e0e0e0', alpha=0.95, linewidth=3))

        # Add main title with ample spacing
        fig.suptitle('GUI Change Analysis - Comprehensive Report',
                     fontsize=32, fontweight='bold', y=0.97)

        # Add subtitle with file information
        ref_name = Path(results['reference_image']).name
        tar_name = Path(results['target_image']).name
        fig.text(0.5, 0.94,
                 f"Reference: {ref_name}  |  Target: {tar_name}",
                 fontsize=20, ha='center', style='italic')

        # Adjust layout with more generous padding at bottom
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ PPT figure saved: {output_path}")
        return output_path

    def create_demo_image(self) -> Image.Image:
        """Create demo image"""
        img = Image.new('RGB', (144, 256), color='#F5F5F5')
        draw = ImageDraw.Draw(img)

        # Draw example components
        draw.rectangle([20, 20, 124, 50], fill='#2196F3', outline='#1565C0', width=2)
        draw.rectangle([22, 80, 122, 110], fill='#FFFFFF', outline='#BDBDBD', width=2)
        draw.rectangle([40, 150, 104, 180], fill='#4CAF50', outline='#388E3C', width=2)
        draw.rectangle([60, 200, 84, 224], fill='#FF9800', outline='#F57C00', width=2)

        return img

    def analyze_gui_changes(self, ref_image_path: str, tar_image_path: str,
                            description: str, output_prefix: str = "gui_analysis"):
        """Main analysis function"""
        print(f"\n{'=' * 60}")
        print("GUI change analysis started")
        print(f"{'=' * 60}")

        start_time = time.time()

        try:
            # Parse detailed changes
            detailed_changes = self.parse_detailed_changes(description)

            # Load images and analyze visual differences
            ref_img, tar_img = self.load_images(ref_image_path, tar_image_path)
            visual_results = self.analyze_visual_difference(ref_img, tar_img)

            # Calculate change intensity stats
            diff_map = visual_results.get('diff_map', None)
            if diff_map is not None:
                visual_results['change_intensity_stats'] = {
                    'low': float((diff_map <= 0.3).mean()),
                    'medium': float(((diff_map > 0.3) & (diff_map <= 0.7)).mean()),
                    'high': float((diff_map > 0.7).mean()),
                    'max_intensity': float(diff_map.max()),
                    'mean_intensity': float(diff_map.mean()),
                    'std_intensity': float(diff_map.std())
                }

            # Build results dictionary
            results = {
                'reference_image': ref_image_path,
                'target_image': tar_image_path,
                'description': description,
                'analysis_time': time.time() - start_time,
                'detailed_changes': detailed_changes,
                'visual_analysis': visual_results,
                'overall_assessment': self.assess_overall_quality(
                    detailed_changes, visual_results
                )
            }

            print(f"\nAnalysis completed in: {results['analysis_time']:.2f} seconds")

            # Create comprehensive figure
            ppt_figure_path = f"{output_prefix}_ppt_comprehensive.png"

            self.create_comprehensive_figure(results, ppt_figure_path)

            print(f"\n{'=' * 60}")
            print("Analysis summary:")
            print(f"  Total changes: {detailed_changes['statistics']['total_changes']}")
            print(f"  Added: {len(detailed_changes['additions'])}")
            print(f"  Removed: {len(detailed_changes['removals'])}")
            print(f"  Moved: {len(detailed_changes['movements'])}")
            print(f"  Validation: {'PASS' if results['overall_assessment']['change_validated'] else 'FAIL'}")
            print(f"\nGenerated file:")
            print(f"  PPT figure: {ppt_figure_path}")
            print(f"{'=' * 60}")

            return {
                'results': results,
                'ppt_figure': ppt_figure_path
            }

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def parse_detailed_changes(self, description: str) -> Dict[str, List[Dict]]:
        """Parse detailed change descriptions"""
        changes = {
            'additions': [],
            'removals': [],
            'movements': [],
            'all_components': []
        }

        # Parse Added
        added_pattern = r'Added (\w+) at position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(added_pattern, description):
            changes['additions'].append({
                'type': comp_type,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                'width': int(x2) - int(x1),
                'height': int(y2) - int(y1),
                'area': (int(x2) - int(x1)) * (int(y2) - int(y1)),
                'change_type': 'addition',
                'description': f"Added {comp_type}"
            })

        # Parse Removed
        removed_pattern = r'Removed (\w+) from position \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, x1, y1, x2, y2 in re.findall(removed_pattern, description):
            changes['removals'].append({
                'type': comp_type,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                'width': int(x2) - int(x1),
                'height': int(y2) - int(y1),
                'area': (int(x2) - int(x1)) * (int(y2) - int(y1)),
                'change_type': 'removal',
                'description': f"Removed {comp_type}"
            })

        # Parse Moved
        moved_pattern = r'(\w+) from \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\) to \((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
        for comp_type, fx1, fy1, fx2, fy2, tx1, ty1, tx2, ty2 in re.findall(moved_pattern, description):
            changes['movements'].append({
                'type': comp_type,
                'from_bbox': [int(fx1), int(fy1), int(fx2), int(fy2)],
                'to_bbox': [int(tx1), int(ty1), int(tx2), int(ty2)],
                'from_center': [(int(fx1) + int(fx2)) // 2, (int(fy1) + int(fy2)) // 2],
                'to_center': [(int(tx1) + int(tx2)) // 2, (int(ty1) + int(ty2)) // 2],
                'change_type': 'movement',
                'description': f"Moved {comp_type}"
            })

        # Merge all components
        changes['all_components'] = (
                changes['additions'] +
                changes['removals'] +
                changes['movements']
        )

        # Statistics
        change_types = {}
        component_types = {}
        for comp in changes['all_components']:
            change_type = comp['change_type']
            comp_type = comp['type']
            change_types[change_type] = change_types.get(change_type, 0) + 1
            component_types[comp_type] = component_types.get(comp_type, 0) + 1

        changes['statistics'] = {
            'change_type_distribution': change_types,
            'component_type_distribution': component_types,
            'total_changes': len(changes['all_components'])
        }

        return changes

    def load_images(self, ref_image_path: str, tar_image_path: str) -> Tuple[Image.Image, Image.Image]:
        """Load reference and target images"""
        if Path(ref_image_path).exists():
            ref_img = Image.open(ref_image_path).convert('RGB')
        else:
            print(f"⚠️ Reference image not found: {ref_image_path}")
            ref_img = self.create_demo_image()

        if Path(tar_image_path).exists():
            tar_img = Image.open(tar_image_path).convert('RGB')
        else:
            print(f"⚠️ Target image not found: {tar_image_path}")
            tar_img = self.create_demo_image()

        return ref_img, tar_img

    def analyze_visual_difference(self, ref_img: Image.Image, tar_img: Image.Image) -> Dict:
        """Analyze visual differences between images"""
        # Resize to same size
        size = (224, 224)
        ref_img_resized = ref_img.resize(size)
        tar_img_resized = tar_img.resize(size)

        # Convert to arrays
        ref_arr = np.array(ref_img_resized).astype(float) / 255.0
        tar_arr = np.array(tar_img_resized).astype(float) / 255.0

        # Calculate difference map
        diff_r = np.abs(ref_arr[:, :, 0] - tar_arr[:, :, 0])
        diff_g = np.abs(ref_arr[:, :, 1] - tar_arr[:, :, 1])
        diff_b = np.abs(ref_arr[:, :, 2] - tar_arr[:, :, 2])

        # Combine channel differences
        diff_combined = (diff_r + diff_g + diff_b) / 3.0

        # Enhance contrast
        diff_enhanced = np.power(diff_combined, 0.7)

        # Normalize
        if diff_enhanced.max() > 0:
            diff_enhanced = diff_enhanced / diff_enhanced.max()

        # Calculate change probability
        change_prob = float(diff_enhanced.mean())

        # Binary mask
        threshold = 0.1
        change_mask = (diff_enhanced > threshold).astype(float)

        # Detect change regions
        change_regions = self.find_change_regions(change_mask)

        return {
            'change_probability': change_prob,
            'change_area_percentage': float(change_mask.mean()),
            'has_change': change_prob > 0.05,
            'change_regions': change_regions,
            'image_size': size,
            'diff_map': diff_enhanced,
            'change_mask': change_mask,
            'threshold_used': threshold
        }

    def find_change_regions(self, change_mask: np.ndarray) -> List[Dict]:
        """Find connected change regions"""
        regions = []

        try:
            from scipy import ndimage
            labeled, num_features = ndimage.label(change_mask > 0.5)

            for i in range(1, num_features + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) > 10:  # Ignore too small regions
                    y_coords, x_coords = coords[:, 0], coords[:, 1]
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()

                    width = x_max - x_min
                    height = y_max - y_min

                    regions.append({
                        'id': i,
                        'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                        'center': [int((x_min + x_max) / 2), int((y_min + y_max) / 2)],
                        'size': [int(width), int(height)],
                        'area': int(width * height),
                        'pixel_count': len(coords)
                    })

            # Sort by area
            regions.sort(key=lambda x: x['area'], reverse=True)
            return regions[:3]  # Return top 3 regions

        except ImportError:
            # If scipy not available, return empty list
            return []

    def assess_overall_quality(self, detailed_changes: Dict, visual_results: Dict) -> Dict:
        """Assess overall quality of changes"""
        stats = detailed_changes['statistics']
        visual_change = visual_results['has_change']

        # Alignment score: described changes vs visual change strength
        described_changes = stats['total_changes']
        visual_strength = visual_results['change_probability']

        if described_changes > 0:
            alignment_score = min(1.0, visual_strength * (1 + min(described_changes, 5) * 0.1))
        else:
            alignment_score = 1.0 if not visual_change else 0.3

        # Overall confidence
        if described_changes > 0:
            overall_confidence = (
                    visual_strength * 0.4 +
                    alignment_score * 0.6
            )
        else:
            overall_confidence = visual_strength

        overall_confidence = np.clip(overall_confidence, 0.0, 1.0)

        # Validation result
        change_validated = (
                visual_change and
                described_changes > 0 and
                alignment_score > 0.6 and
                overall_confidence > 0.5
        )

        return {
            'alignment_score': float(alignment_score),
            'overall_confidence': float(overall_confidence),
            'described_changes_count': described_changes,
            'visual_change_detected': visual_change,
            'change_validated': change_validated,
            'summary': self.generate_quality_summary(
                stats, visual_results, change_validated
            )
        }

    def generate_quality_summary(self, stats: Dict, visual_results: Dict, validated: bool) -> str:
        """Generate quality summary text"""
        if validated:
            return f"Validation passed: {stats['total_changes']} changes align with visual detection"

        if stats['total_changes'] == 0 and not visual_results['has_change']:
            return "No changes detected"

        if stats['total_changes'] == 0 and visual_results['has_change']:
            return "Visual changes detected but no description"

        if stats['total_changes'] > 0 and not visual_results['has_change']:
            return "Changes described but no significant visual changes"

        return "Changes require further verification"


def main():
    """Main function"""
    analyzer = PPTGUIAnalyzer()

    # Test case
    test_case = {
        'name': 'Login Screen Component Change Analysis',
        'ref_image': "/home/common-dir/data/gui/settings/23319.png",
        'tar_image': "/home/common-dir/data/gui/settings/57622.png",
        'description': """Added TextView at position (0, 83, 144, 188); Added View at position (0, 9, 144, 38); Added SwitchMain at position (0, 37, 72, 81); Added SwitchSlider at position (72, 37, 144, 81); TextView from (26, 130, 40, 137) to (0, 83, 144, 188); TextView from (26, 111, 49, 118) to (0, 83, 144, 188)"""
    }

    print(f"\nTest case: {test_case['name']}")
    print(f"{'=' * 60}")

    # Run analysis
    result = analyzer.analyze_gui_changes(
        test_case['ref_image'],
        test_case['tar_image'],
        test_case['description'],
        output_prefix="login_screen_ppt"
    )

    if 'error' not in result:
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed: {result['error']}")

    print(f"\n{'=' * 60}")
    print("GUI change analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()