import os

from moviepy.editor import VideoFileClip
# making all videos not more than 10s 

fake_dir = r'/home/kundan/Documents/DeepLearning/ml_project/data_save/fake'
real_dir = r'/home/kundan/Documents/DeepLearning/ml_project/data_save/real'


def video_trimmer(folder_path):
    count=0
    output_folder = folder_path+"_t"
    for file in os.listdir(folder_path):
        print(f"file trimming in path : {folder_path}")
        if count > 1000:break
        if file.endswith('.mp4'):
            input_f = os.path.join(folder_path,file)
            output_f = os.path.join(output_folder,file)
            try:
                with VideoFileClip(input_f) as video:
                    trimmed_clip = video.subclip(0, 8)  # Trim the first 10 seconds
                    trimmed_clip.write_videofile(output_f, codec='libx264')
                    trimmed_clip.close()  # Explicitly close the clip
                    count+=1
            except Exception as e:
                print(f"Error processing {file}: {e}")

def main():
    fake_dir = r'/home/kundan/Documents/DeepLearning/ml_project/data_save/fake'
    real_dir = r'/home/kundan/Documents/DeepLearning/ml_project/data_save/real'
    video_trimmer(fake_dir)
    video_trimmer(real_dir)


main()
# renamed them to fake_t and real_t to fake and real and moved to FAKE-VIDEO-DETECTOR folder