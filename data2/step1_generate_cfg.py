# encoding=utf-8
import os, sys
import glob
import argparse
from multiprocessing import Pool
from functools import partial
import subprocess


def parse_options():
    parser = argparse.ArgumentParser(description='Extracting Cpgs.')
    parser.add_argument('-i', '--input', help='The dir path of input', type=str,
                        default='/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs/bins')
    parser.add_argument('-o', '--output', help='The dir path of output', type=str,
                        default='/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/cfgs/pdgs')
    parser.add_argument('-t', '--type', help='The type of procedures: parse or export', type=str, default='export')
    parser.add_argument('-r', '--repr', help='The type of representation: pdg or lineinfo_json', type=str,
                        default='cfg')
    args = parser.parse_args()
    return args


def joern_parse(file, outdir):
    name = file.split('/')[-1].split('.')[0]
    print(' ----> now processing: ', name)
    out = outdir + name + '.bin'
    if os.path.exists(out):
        return
    os.environ['file'] = str(file)
    os.environ['out'] = str(out)  # parse后的文件名与source文件名称一致
    # os.system('sh joern-parse $file --language c --out $out')
    os.system('sh joern-parse $file --out $out')


def joern_export(bin, outdir, repr):
    name = bin.split('/')[-1].split('.')[0]
    out = os.path.join(outdir, name)
    print(' ----> now processing: ', name)
    os.environ['bin'] = str(bin)
    os.environ['out'] = str(out)

    if repr == 'pdg':
        os.system('sh joern-export $bin' + " --repr " + "pdg" + ' --out $out')
        try:
            pdg_list = os.listdir(out)
            for pdg in pdg_list:
                if pdg.startswith("0-pdg"):
                    file_path = os.path.join(out, pdg)
                    os.system("mv " + file_path + ' ' + out + '.dot')
                    os.system("rm -rf " + out)
                    break
        except:
            pass
    elif repr == 'cfg':
        os.system('sh joern-export $bin' + " --repr " + "cfg" + ' --out $out')
        file_path = os.path.join(out, "0-cfg.dot")
        if os.path.exists(file_path):
            os.system("mv " + file_path + ' ' + out + '.dot')
            os.system("rm -rf " + out)
        else:
            print(file_path, "can not generate CFG")
            with open("/data/bhtian2/win_linux_mapping/three-fusion/data2/our26/no_train.txt", 'a') as f:
                os.system("rm -rf " + out)
                f.write(file_path)
    else:
        pwd = os.getcwd()
        if out[-4:] != 'json':
            out += '.json'
        joern_process = subprocess.Popen(["./joern"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True,
                                         encoding='utf-8')
        import_cpg_cmd = f"importCpg(\"{bin}\")\r"
        script_path = f"{pwd}/graph-for-funcs.sc"
        run_script_cmd = f"cpg.runScript(\"{script_path}\").toString() |> \"{out}\"\r"  # json
        cmd = import_cpg_cmd + run_script_cmd
        ret, err = joern_process.communicate(cmd)
        print(ret, err)



def main():
    joern_path = '/data/bhtian2/bin/joern-cli'
    os.chdir(joern_path)
    args = parse_options()

    type = args.type
    repr = args.repr

    input_path = args.input
    output_path = args.output

    if input_path[-1] == '/':
        input_path = input_path
    else:
        input_path += '/'

    if output_path[-1] == '/':
        output_path = output_path
    else:
        output_path += '/'

    pool_num = 96
    pool = Pool(pool_num)

    if type == 'parse':
        # files = get_all_file(input_path)
        files = glob.glob(input_path + '*.c')
        pool.map(partial(joern_parse, outdir=output_path), files)

    elif type == 'export':
        bins = glob.glob(input_path + '*.bin')
        if repr == 'pdg':
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)
        else:
            pool.map(partial(joern_export, outdir=output_path, repr=repr), bins)

    else:
        print('Type error!')


if __name__ == '__main__':
    main()
