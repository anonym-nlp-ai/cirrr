import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


#####################################################################################
### Adding doc count/percentage over outer refinement iterations intervals over `200` documents
#######################################################################################
# Define the intervals and corresponding percentages for the 4 models
intervals =                             ['[1,2]', '[3,4]', '[5,6]']

cir3_doc_count_percent =                [  4.0  ,   57.0 ,   39.0] 
cir3_agent_only_doc_count_percent =     [  5.5  ,   44.5 ,   50.0 ]
cir3_vendi_only_doc_count_percent =     [  0.5  ,   14.5 ,   85.0 ]
cir3_random_reject_doc_count_percent =  [ 73.0  ,   17.0 ,   10.0 ]

#####################################################################################
### Adding `Comp` scores over outer refinement iterations intervals over `200` documents
#######################################################################################
# TODO: ADD values to the table
# THESE ARE AVG OF FINAL SCORES. WE SAVED COMP SCORES AFTER PROCESS COMPLETE <--------------
# intervals =                                  ['[1,2]', '[3,4]', '[5,6]']
cir3_comp_scores_per_interval =                [0.9412 , 0.9674 ,  0.9499] # average score: 0.9528
cir3_agent_only_comp_scores_per_interval =     [0.9163 , 0.9221 ,  0.9285] # average score: 0.9223
cir3_vendi_only_comp_scores_per_interval =     [0.9096 , 0.8876 ,  0.8709] # average score: 0.8893
cir3_random_reject_comp_scores_per_interval =  [0.8165 , 0.8343 ,  0.7993] # average score: 0.8167

print(sum(cir3_comp_scores_per_interval) / len(cir3_comp_scores_per_interval))
print(sum(cir3_agent_only_comp_scores_per_interval) / len(cir3_agent_only_comp_scores_per_interval))
print(sum(cir3_vendi_only_comp_scores_per_interval) / len(cir3_vendi_only_comp_scores_per_interval))
print(sum(cir3_random_reject_comp_scores_per_interval) / len(cir3_random_reject_comp_scores_per_interval))

#####################################################################################
### Adding `Comp` scores for documents which were processed with `k=5` outer refinement iterations over `200` documents
#######################################################################################
# The final score of these lists should converge to the ones from the intervals [5, 6]
cir3_avg_comp_scores_for_doc_with_outer_5 =                [0.7801, 0.8795, 0.9391, 0.9437, 0.9573] #  doc count percentage: 24.5%
cir3_agent_only_avg_comp_scores_for_doc_with_outer_5 =     [0.7819, 0.8553, 0.9180, 0.9193, 0.9291] #  doc count percentage: 29.0%
cir3_vendi_only_avg_comp_scores_for_doc_with_outer_5 =     [0.7923, 0.8333, 0.8554, 0.8622, 0.8770] #  doc count percentage: 32.0%
cir3_random_avg_comp_scores_for_doc_with_outer_5 =         [0.7811, 0.7886, 0.7909, 0.7991, 0.7975] #  doc count percentage: 7.0%
cir3_doc_count_percent_with_outer_5 =              24.5
cir3_agent_only_doc_count_percent_with_outer_5 =    29.0
cir3_vendi_only_doc_count_percent_with_outer_5 =    32.0
cir3_random_reject_doc_count_percent_with_outer_5 = 7.0

# Combine all percentages into a single list
all_percentages = cir3_doc_count_percent + cir3_agent_only_doc_count_percent + cir3_vendi_only_doc_count_percent + cir3_random_reject_doc_count_percent

# Calculate the maximum percentage value
max_percentage = max(all_percentages)

# Set the positions and width for the bars
x = range(len(intervals))
width = 0.15  # Width of the bars

# Define hatch patterns for each model
hatches = ['xx', '||', '..', '//']

# colors = sns.color_palette("Greys", n_colors=3)
# colors = ['#add8e6', '#fffdd0', '#f5f5dc']
# Define the colors for light gray, light blue, and beige
colors = ['#d3d3d3', '#add8e6', '#f5f5dc', '#c3aed6']  # Light gray, light blue, beige, light purple

# Create a comprehensive visualization with multiple approaches
# First row: Document Distribution and Comprehensiveness Scores
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Document Distribution (Original)
x = range(len(intervals))
width = 0.15
hatches = ['xx', '||', '..', '//']
colors = ['#d3d3d3', '#add8e6', '#f5f5dc', '#c3aed6']

bars1 = ax1.bar([p - width for p in x], cir3_doc_count_percent, width=width, label='CIR3', hatch=hatches[0], edgecolor='black', color=colors[0])
bars2 = ax1.bar(x, cir3_agent_only_doc_count_percent, width=width, label='CIR3 Intrinsic Only', hatch=hatches[1], edgecolor='black', color=colors[1])
bars3 = ax1.bar([p + width for p in x], cir3_vendi_only_doc_count_percent, width=width, label='CIR3 Vendi Only', hatch=hatches[2], edgecolor='black', color=colors[2])
bars4 = ax1.bar([p + width * 2 for p in x], cir3_random_reject_doc_count_percent, width=width, label='CIR3 Random Rejection', hatch=hatches[3], edgecolor='black', color=colors[3])

ax1.set_xticks([p + width/2 for p in x])
ax1.set_xticklabels(intervals)
ax1.set_ylabel('Processed Documents (%)', fontsize=12)
ax1.set_title('Document Distribution Across Cycle Ranges')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Comprehensiveness Scores
# Use dots/points to show discrete scores for each interval
colors_comp = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
comp_data = [cir3_comp_scores_per_interval, cir3_agent_only_comp_scores_per_interval, 
             cir3_vendi_only_comp_scores_per_interval, cir3_random_reject_comp_scores_per_interval]
labels_comp = ['CIR3', 'CIR3 Intrinsic Only', 'CIR3 Vendi Only', 'CIR3 Random Rejection']
markers = ['o', 's', '^', 'D']

for i, (scores, label, color, marker) in enumerate(zip(comp_data, labels_comp, colors_comp, markers)):
    # Plot dots for each interval
    ax2.scatter(x, scores, s=120, color=color, marker=marker, alpha=0.8, edgecolor='black', linewidth=1, label=label)
    # Add score value as text with white background for readability
    for j, score in enumerate(scores):
        ax2.annotate(f'{score:.4f}', (j, score), xytext=(0, 8), textcoords='offset points', 
                    ha='center', fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

ax2.set_xticks(x)
ax2.set_xticklabels(intervals)
ax2.set_ylabel('Comprehensiveness Score', fontsize=12)
ax2.set_title('Comprehensiveness Scores Across Cycle Ranges')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0.75, 1.0)

plt.tight_layout()
plt.show()

# Second row: Processing Speed vs Comprehensiveness and Heatmap
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 3: Processing Speed vs Comprehensiveness Scatter
# Calculate processing speed as inverse of average cycles (lower cycles = faster processing)
# This represents how quickly documents converge to final state
processing_speed_cir3 = 1 / (sum([i * p/100 for i, p in enumerate(cir3_doc_count_percent, 1)]))
processing_speed_agent = 1 / (sum([i * p/100 for i, p in enumerate(cir3_agent_only_doc_count_percent, 1)]))
processing_speed_vendi = 1 / (sum([i * p/100 for i, p in enumerate(cir3_vendi_only_doc_count_percent, 1)]))
processing_speed_random = 1 / (sum([i * p/100 for i, p in enumerate(cir3_random_reject_doc_count_percent, 1)]))

# Alternative: Direct average cycles (more intuitive)
avg_cycles_cir3 = sum([i * p/100 for i, p in enumerate(cir3_doc_count_percent, 1)])
avg_cycles_agent = sum([i * p/100 for i, p in enumerate(cir3_agent_only_doc_count_percent, 1)])
avg_cycles_vendi = sum([i * p/100 for i, p in enumerate(cir3_vendi_only_doc_count_percent, 1)])
avg_cycles_random = sum([i * p/100 for i, p in enumerate(cir3_random_reject_doc_count_percent, 1)])

avg_scores = [
    sum(cir3_comp_scores_per_interval) / len(cir3_comp_scores_per_interval),
    sum(cir3_agent_only_comp_scores_per_interval) / len(cir3_agent_only_comp_scores_per_interval),
    sum(cir3_vendi_only_comp_scores_per_interval) / len(cir3_vendi_only_comp_scores_per_interval),
    sum(cir3_random_reject_comp_scores_per_interval) / len(cir3_random_reject_comp_scores_per_interval)
]

# Use average cycles for x-axis (more intuitive)
processing_metrics = [avg_cycles_cir3, avg_cycles_agent, avg_cycles_vendi, avg_cycles_random]
labels = ['CIR3', 'CIR3 Intrinsic Only', 'CIR3 Vendi Only', 'CIR3 Random Rejection']
colors_scatter = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
offsets_scatter = [(7, -7), (5, 5), (-50, 7), (5, 5)]  # left-up, right-up, left-down, right-down
for i, (cycles, score, label, color) in enumerate(zip(processing_metrics, avg_scores, labels, colors_scatter)):
    ax3.scatter(cycles, score, s=100, color=color, label=label, alpha=0.7)
    ax3.annotate(f'{label}\n({cycles:.2f} cycles)', (cycles, score), xytext=offsets_scatter[i], textcoords='offset points', fontsize=8)

ax3.set_xlabel('Average Processing Cycles')
ax3.set_ylabel('Average Comprehensiveness Score')
ax3.set_title('Processing Speed vs Comprehensiveness Trade-off')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Comprehensiveness Heatmap
performance_data = [
    cir3_comp_scores_per_interval,
    cir3_agent_only_comp_scores_per_interval,
    cir3_vendi_only_comp_scores_per_interval,
    cir3_random_reject_comp_scores_per_interval
]

im = ax4.imshow(performance_data, cmap='RdYlGn', aspect='auto', vmin=0.75, vmax=1.0)
ax4.set_xticks(range(len(intervals)))
ax4.set_xticklabels(intervals)
ax4.set_yticks(range(len(labels)))
ax4.set_yticklabels(labels)
ax4.set_xlabel('Cycle Ranges')
ax4.set_title('Comprehensiveness Heatmap')
plt.colorbar(im, ax=ax4, label='Comprehensiveness Score')

# Add value annotations to heatmap
for i in range(len(performance_data)):
    for j in range(len(performance_data[0])):
        text = ax4.text(j, i, f'{performance_data[i][j]:.4f}', 
                       ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.show()

# Print the actual values for clarity
print("Average Processing Cycles:")
print(f"CIR3: {avg_cycles_cir3:.2f} cycles")
print(f"CIR3 Intrinsic Only: {avg_cycles_agent:.2f} cycles") 
print(f"CIR3 Vendi Only: {avg_cycles_vendi:.2f} cycles")
print(f"CIR3 Random Rejection: {avg_cycles_random:.2f} cycles")
print("\nProcessing Speed (1/cycles):")
print(f"CIR3: {processing_speed_cir3:.4f}")
print(f"CIR3 Intrinsic Only: {processing_speed_agent:.4f}")
print(f"CIR3 Vendi Only: {processing_speed_vendi:.4f}")
print(f"CIR3 Random Rejection: {processing_speed_random:.4f}")

# Alternative: Simple dual-axis approach (cleaner version)
fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

# Bars for document distribution
bars1 = ax1.bar([p - width for p in x], cir3_doc_count_percent, width=width, label='CIR3', hatch=hatches[0], edgecolor='black', color=colors[0], alpha=0.7)
bars2 = ax1.bar(x, cir3_agent_only_doc_count_percent, width=width, label='CIR3 Intrinsic Only', hatch=hatches[1], edgecolor='black', color=colors[1], alpha=0.7)
bars3 = ax1.bar([p + width for p in x], cir3_vendi_only_doc_count_percent, width=width, label='CIR3 Vendi Only', hatch=hatches[2], edgecolor='black', color=colors[2], alpha=0.7)
bars4 = ax1.bar([p + width * 2 for p in x], cir3_random_reject_doc_count_percent, width=width, label='CIR3 Random Rejection', hatch=hatches[3], edgecolor='black', color=colors[3], alpha=0.7)

# Lines for performance scores
# Use dots/points to show discrete scores for each interval, positioned above corresponding bars
colors_comp_dual = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
comp_data_dual = [cir3_comp_scores_per_interval, cir3_agent_only_comp_scores_per_interval, 
                  cir3_vendi_only_comp_scores_per_interval, cir3_random_reject_comp_scores_per_interval]
labels_comp_dual = ['CIR3 Comprehensiveness', 'CIR3 Intrinsic Only Comprehensiveness', 
                    'CIR3 Vendi Only Comprehensiveness', 'CIR3 Random Rejection Comprehensiveness']
markers_dual = ['o', 's', '^', 'D']

# Position dots above their corresponding bars
bar_positions = [[p - width for p in x], x, [p + width for p in x], [p + width * 2 for p in x]]

for i, (scores, label, color, marker, bar_pos) in enumerate(zip(comp_data_dual, labels_comp_dual, colors_comp_dual, markers_dual, bar_positions)):
    # Plot dots for each interval, positioned above corresponding bars
    ax2.scatter(bar_pos, scores, s=120, color=color, marker=marker, alpha=0.8, edgecolor='black', linewidth=1, label=label)
    # Add score value as text with white background for readability
    for j, (score, pos) in enumerate(zip(scores, bar_pos)):
        ax2.annotate(f'{score:.4f}', (pos, score), xytext=(0, 8), textcoords='offset points', 
                    ha='center', fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

ax1.set_xticks([p + width/2 for p in x])
ax1.set_xticklabels(intervals)
ax1.set_ylabel('Processed Documents (%)', fontsize=12)
ax2.set_ylabel('Comprehensiveness Score', fontsize=12, color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')
ax2.set_ylim(0.75, 1.0)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.15, 0.5))

ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

#####################################################################################
### NEW FIGURE: Comprehensiveness scores for documents with exactly 5 outer iterations
#####################################################################################

# Create figure for documents processed with exactly 5 outer refinement iterations
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Document count percentage for documents with 5 outer iterations
doc_counts_5_iterations = [
    cir3_doc_count_percent_with_outer_5,
    cir3_agent_only_doc_count_percent_with_outer_5,
    cir3_vendi_only_doc_count_percent_with_outer_5,
    cir3_random_reject_doc_count_percent_with_outer_5
]

model_names = ['CIR3', 'CIR3 Intrinsic Only', 'CIR3 Vendi Only', 'CIR3 Random Rejection']
# colors_5_iter = ['darkred', 'darkblue', 'darkgreen', 'darkorange']
colors_5_iter = ['#f8c4c4', 'lightblue', '#d2f8d2', '#ffc04d']
hatches_5_iter = ['xx', '||', '..', '//']

bars_5_iter = ax5.bar(model_names, doc_counts_5_iterations, color=colors_5_iter, alpha=0.7, 
                      edgecolor='black', hatch=hatches_5_iter)
ax5.set_ylabel('Document Count Percentage (%)')
ax5.set_title('Documents Requiring Exactly 5 Outer Refinement Iterations')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars_5_iter, doc_counts_5_iterations):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Comprehensiveness scores progression over 5 iterations
iteration_numbers = [1, 2, 3, 4, 5]

line1_5iter = ax6.plot(iteration_numbers, cir3_avg_comp_scores_for_doc_with_outer_5, 'o-', 
                       color='darkred', linewidth=2, markersize=8, label='CIR3')
line2_5iter = ax6.plot(iteration_numbers, cir3_agent_only_avg_comp_scores_for_doc_with_outer_5, 's-', 
                       color='darkblue', linewidth=2, markersize=8, label='CIR3 Intrinsic Only')
line3_5iter = ax6.plot(iteration_numbers, cir3_vendi_only_avg_comp_scores_for_doc_with_outer_5, '^-', 
                       color='darkgreen', linewidth=2, markersize=8, label='CIR3 Vendi Only')
line4_5iter = ax6.plot(iteration_numbers, cir3_random_avg_comp_scores_for_doc_with_outer_5, 'D-', 
                       color='darkorange', linewidth=2, markersize=8, label='CIR3 Random Rejection')

ax6.set_xlabel('Outer Refinement Iteration')
ax6.set_ylabel('Comprehensiveness Score')
ax6.set_title('Comprehensiveness Score Progression for Documents Requiring 5 Iterations')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim(0.75, 1.0)

# Add final score annotations
final_scores = [
    cir3_avg_comp_scores_for_doc_with_outer_5[-1],
    cir3_agent_only_avg_comp_scores_for_doc_with_outer_5[-1],
    cir3_vendi_only_avg_comp_scores_for_doc_with_outer_5[-1],
    cir3_random_avg_comp_scores_for_doc_with_outer_5[-1]
]

# # annotates the final score (iteration 5) for each model:
# for i, (score, name) in enumerate(zip(final_scores, model_names)):
#     ax6.annotate(f'{score:.4f}', (5, score), xytext=(5.2, score), 
#                 textcoords='data', fontsize=8, fontweight='bold',
#                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
# Annotate all 5 scores for each model
all_scores = [
    cir3_avg_comp_scores_for_doc_with_outer_5,
    cir3_agent_only_avg_comp_scores_for_doc_with_outer_5,
    cir3_vendi_only_avg_comp_scores_for_doc_with_outer_5,
    cir3_random_avg_comp_scores_for_doc_with_outer_5
]

# for model_scores in all_scores:
#     for iter_num, score in enumerate(model_scores, 1):
#         ax6.annotate(f'{score:.4f}', (iter_num, score), xytext=(0, 8), textcoords='offset points',
#                      ha='center', fontsize=8, fontweight='bold', color='black',
#                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# offsets = [(-25, 8), (30, 8), (-25, -18), (30, -18)]  # left-up, right-up, left-down, right-down
offsets = [(30, -10), (30, 8), (-25, -4), (-25, 0)]  # left-up, right-up, left-down, right-down


# facecolors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
facecolors = ['#f8c4c4', 'lightblue', '#d2f8d2', '#ffc04d']
for model_idx, model_scores in enumerate(all_scores):
    for iter_num, score in enumerate(model_scores, 1):
        if iter_num == 1:
            xytext = offsets[model_idx]
            ax6.annotate(
                f'{score:.4f}', (iter_num, score), xytext=xytext, textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolors[model_idx], alpha=0.8)
            )
        else:
            xytext = (0, 8)
            ax6.annotate(
                f'{score:.4f}', (iter_num, score), xytext=xytext, textcoords='offset points',
                ha='center', fontsize=8, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolors[model_idx], alpha=0.8)
            )


plt.tight_layout()
plt.show()

# Print summary statistics for the 5-iteration documents
print("\n" + "="*60)
print("DOCUMENTS REQUIRING EXACTLY 5 OUTER REFINEMENT ITERATIONS")
print("="*60)
print("Document Count Percentages:")
for name, count in zip(model_names, doc_counts_5_iterations):
    print(f"  {name}: {count}%")

print("\nFinal Comprehensiveness Scores (after 5 iterations):")
for name, score in zip(model_names, final_scores):
    print(f"  {name}: {score:.4f}")

print("\nScore Improvement (from iteration 1 to 5):")
for name, scores in zip(model_names, [cir3_avg_comp_scores_for_doc_with_outer_5, 
                                     cir3_agent_only_avg_comp_scores_for_doc_with_outer_5,
                                     cir3_vendi_only_avg_comp_scores_for_doc_with_outer_5,
                                     cir3_random_avg_comp_scores_for_doc_with_outer_5]):
    improvement = scores[-1] - scores[0]
    print(f"  {name}: {improvement:.4f} ({scores[0]:.4f} → {scores[-1]:.4f})")