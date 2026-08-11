import csv

with open("Tweets.txt", "r", encoding="utf-8") as file_in, open("data.csv", "w", newline="", encoding="utf-8") as file_out:
    writer = csv.writer(file_out)
    writer.writerow(["text", "label"])
    i = 0
    for line in file_in:
        i += 1
        parts = line.strip().rsplit("\t", 1) 
        if len(parts) == 2:
            writer.writerow(parts)
        else:
            print("⚠️ Error:", line)
            print(i)
