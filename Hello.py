import luigi
import subprocess

class Scrapper(luigi.Task):
    def requires(self):
        return None
    def output(self):
        return None #luigi.LocalTarget("/home/maximiliano/Documents/luigi-hello-world/Scrapeo/Base.json")
    def run(self):
        subprocess.call(['./Update.sh'])

if __name__ == '__main__':
    luigi.run()
