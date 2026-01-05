import time

def convert_time(t):
    t = int(t)
    hrs, rem = divmod(t, 3600)
    mins, secs = divmod(rem, 60)
    return hrs, mins, secs

class ProgressTracker:

    def __init__(self):

        self.epoch_count = 0
        self.iter_count = 0

        self.current_c_loss = []
        self.current_r_loss = []
        self.current_total_loss = []

        self.epochs_c_loss = []
        self.epochs_r_loss = []
        self.eochs_total_loss = []

    def record_losses(self, c_loss, r_loss, loss):
        self.iter_count += 1

        self.current_c_loss.append(float(c_loss))
        self.current_r_loss.append(float(r_loss))
        self.current_total_loss.append(float(loss))

        if self.iter_count % 1 == 0:
            t = time.time() - self.epoch_timer
            h, m, s = convert_time(t)
            p = f'\t\t{self.iter_count} iterations | '
            p += f'Last classification loss {c_loss:.3f} | '
            p += f'Last regression loss {r_loss:.3f} | '
            p += f'Last total loss {loss:.3f} | '
            p += f'Epoch time {h}H {m}M {s}S'
            print(p)  

    def start_timer(self):
        self.timer = time.time()

    def start_epoch(self):
        self.epoch_count += 1
        self.epoch_timer = time.time()
        print(f'\tEpoch {self.epoch_count}:\n')

    def end_epoch(self):

        t_training = time.time() - self.timer
        t_epoch = time.time() - self.epoch_timer

        h_train, m_train, s_train = convert_time(t_training)
        h_epoch, m_epoch, s_epoch = convert_time(t_epoch)


        avg_c_loss = sum(self.current_c_loss) / len(self.current_c_loss)
        avg_r_loss = sum(self.current_r_loss) / len(self.current_r_loss)
        avg_total_loss = sum(self.current_total_loss) / len(self.current_total_loss)

        self.current_c_loss = []
        self.current_r_loss = []
        self.current_total_loss = []
        self.iter_count = 0

        p = f'\tFinished:\n\tEpoch {self.epoch_count} | '
        p += f'Num iters {self.iter_count} | '
        p += f'Average class loss {avg_c_loss:.3f} | '
        p += f'Average regression loss {avg_r_loss:.3f} | '
        p += f'Average total loss {avg_total_loss:.3f} | '
        p += f'Epoch time {h_epoch}H {m_epoch}M {s_epoch}S | '
        p += f'Training time {h_train}H {m_train}M {s_train}S'
        print(p)