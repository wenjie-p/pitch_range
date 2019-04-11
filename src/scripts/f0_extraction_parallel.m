function [] =  wenjie_processing(scp)

steps = length(scp);

%f0floor = 0;
%f0ceil  = 0;

for i = 1: steps

    fin = scp{i}{1};
%    f0floor = scp{i}{2};
%    f0ceil = scp{i}{3};

    des = scp{i}{2};
    fout = scp{i}{3};
    f0floor = scp{i}{4};
    f0ceil = scp{i}{5};
%    gender = scp{i}{4};
%    if gender == 'M'
%        f0floor = 50;
%        f0ceil = 300;
%    else
%        f0floor = 75;
%        f0ceil  = 500;
%    end

    f0raw = 0;
    try
        [x, fs] = audioread(fin);
        %f0raw = MulticueF0v14(x, fs, f0floor, f0ceil);
        f0raw = MulticueF0v14(x, fs );
    catch
        fprintf('processing file: %s failed\n', fin);
        %fprintf(ME);
        continue;
    end

%    disp(size(f0raw));
%    disp(des);
%    disp(fout);
    if ~ exist(des, 'dir')
        mkdir(des);
    end
    fp = fopen(fout, 'w');
    fprintf(fp, '%.7f\n', f0raw);
    fclose(fp);
    fprintf('processing file: %s successfully\n.', fin);

end

end
