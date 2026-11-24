from train_policy import train_mc
from matplotlib import pyplot as plt
import time

for er in range(10):
    for alpha in range(10):
        start = time.time()
        exploration_rate = er/10
        a = alpha/10
        log = train_mc(5,100,exploration_rate,a,.9)
        scores = []
        unseen = []
        for l in log:
            scores.append(l['avg_score'])
            unseen.append(l['percent_unseen'])
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(scores)
        ax1.set_title('average score vs iteration')
        ax2.plot(unseen)
        ax2.set_title('percent unseen vs iteration')
        title = f'exploration rate={exploration_rate},a={a}'
        fig.suptitle(title)
        plt.savefig(f'./container_files/test_plots/{title}.png')
        plt.close(fig)
        print(f'{title}: {(time.time()-start)/60} Minutes')