box_crab_count = {}

# Process detections
for *xyxy, conf, cls in reversed(det):
    label = f'{names[int(cls)]} {conf:.2f}'
    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

    if names[int(cls)] == 'Crab':
        # Get the coordinates of the current box
        box_coords = (xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1])

        if box_coords not in box_crab_count:
            box_crab_count[box_coords] = 0

        # Increment the count for the current box
        box_crab_count[box_coords] += 1

# Check if any box has 2 or more crabs and label them as "Moulted"
for box_coords, crab_count in box_crab_count.items():
    if crab_count >= 2:
        x, y, w, h = box_coords
        moulted_label = 'Moulted'
        cv2.putText(im0, moulted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)