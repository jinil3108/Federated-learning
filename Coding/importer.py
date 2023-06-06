import random
import shutil
import os
import util

if __name__ == '__main__':
    # Dividing the images into equal ratio between clients.
    # Hence, it becomes a balance IID (Identically and Independent Distributed Data) condition.

    # Taking out the test datasets from the table.
    if os.path.isdir(os.path.join(util.root_dir, util.source_dir[1])):
        os.mkdir(os.path.join(util.root_dir, 'test'))

        for c in util.source_dir:
            os.mkdir(os.path.join(util.root_dir, 'test', c))

        for c in util.source_dir:
            images = []
            for x in os.listdir(os.path.join(util.root_dir, c)):
                if x.lower().endswith('png'):
                    images.append(x)

            print(len(images)//util.num_clients)

            selected_images = random.sample(images, len(images) // (util.num_clients+1))

            print(selected_images)
            for image in selected_images:
                source_path = os.path.join(util.root_dir, c, image)
                target_path = os.path.join(util.root_dir, 'test', c, image)
                shutil.move(source_path, target_path)

    # Creating a folder named as train.
    os.mkdir(os.path.join(util.root_dir, 'train'))

    # Transforming the whole data sets into number of clients.
    for i in range(util.num_clients):
        os.mkdir(os.path.join(util.root_dir, 'train', str(i+1)))
        os.mkdir(os.path.join(util.root_dir, 'train', str(i+1), 'COVID'))
        os.mkdir(os.path.join(util.root_dir, 'train', str(i+1), 'non-COVID'))

        # Covid Images dividing into the random samples.
        images = []
        for x in os.listdir(os.path.join(util.root_dir, 'COVID')):
            if x.lower().endswith('png'):
                images.append(x)

        selected_images = random.sample(images, util.total_covid_images // (util.num_clients+1))

        for image in selected_images:
            source_path = os.path.join(util.root_dir, 'COVID', image)
            target_path = os.path.join(util.root_dir, 'train', str(i+1), 'COVID', image)
            shutil.move(source_path, target_path)

        # Non - Covid Images dividing into the random samples.
        images = []
        for x in os.listdir(os.path.join(util.root_dir, 'non-COVID')):
            if x.lower().endswith('png'):
                images.append(x)

        selected_images = random.sample(images, util.total_non_covid_images // (util.num_clients+1))

        for image in selected_images:
            source_path = os.path.join(util.root_dir, 'non-COVID', image)
            target_path = os.path.join(util.root_dir, 'train', str(i + 1), 'non-COVID', image)
            shutil.move(source_path, target_path)

