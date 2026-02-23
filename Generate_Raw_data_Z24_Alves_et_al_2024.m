% Alves, V., Cury, A. & Fortes, R. (2024).

clear;clc;
%Function to collect data data of Z24 Bridge
% Select sensors from the predetermined mesh by removing those unwanted sensors 

%% Creates square mesh of 4 sensors/setup (Pre-determined reduced mesh of 4x9=36 sensors to facilitate sensor selection)
%In avt (Ambient Vibration)
name_sens=['300V';'302V';'305V';'307V';'310V';'312V';'315V';'317V';'320V';'322V';'325V';'327V';'330V';'332V';'335V';'337V';'340V';'342V';'100V';'102V';'105V';'107V';'110V';'112V';'115V';'117V';'120V';'122V';'125V';'127V';'130V';'132V';'135V';'137V';'140V';'142V'];
% from this 36 sensor mesh, being these as a vector:
%[300V, 302V, 100V, 102V,       (setup 1)
% 305V, 307V, 105V, 107V,       (setup 2)
% 310V, 312V, 110V, 112V,       (setup 3)
% 315V, 317V, 115V, 117V,       (setup 4)
% 320V, 322V, 120V, 122V,       (setup 5)
% 325V, 327V, 125V, 127V,       (setup 6)
% 330V, 332V, 130V, 132V,       (setup 7)
% 335V, 337V, 135V, 137V,       (setup 8)
% 340V, 342V, 140V, 142V]       (setup 9)

% Then, 7 sensor will be selected after collecting, being: [105V, 110V, 115V, 120V, 125V, 130V, 135V].
% So, it will be removed the position:
% [1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36].
%% Collecting data
for j=1:9
    disp("------------------------------------------------------------------")
    disp("seção"+j)
    load("pdt_01-08/02/avt/02setup0"+j+".mat") % adapt this path to your folder
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
           disp("Sensor position (collumn) in generated matrix: "+int2str(4*(j-1)+1))
    disp(name_sens(2*j-1,:))
    end

    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
                  disp("Sensor position (collumn) in generated matrix: "+int2str(4*(j-1)+2))
    disp(name_sens(2*j,:))
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
                  disp("Sensor position (collumn) in generated matrix: "+int2str(4*(j-1)+3))
    disp(name_sens(2*j+17,:))
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
                  disp("Sensor position (collumn) in generated matrix: "+int2str(4*(j-1)+4))
    disp(name_sens(2*j+18,:))
    end
    end
end
end
Z24_d0=sens;
Z24_d0(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d0","Z24_d0")
disp("D0 - concluded")
%----------------------------------------------------------------
for j=1:9
    disp("seção"+j)
    load("pdt_01-08/03/avt/03setup0"+j+".mat")
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
    end
    end
end
end
Z24_d1=sens;
Z24_d1(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d1","Z24_d1")
disp("D1 - concluded")
%----------------------------------------------------------------
for j=1:9
    disp("seção"+j)
    load("pdt_01-08/04/avt/04setup0"+j+".mat")
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
    end
    end
end
end
Z24_d2=sens;
Z24_d2(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d2","Z24_d2")
disp("D2 - concluded")
%----------------------------------------------------------------
for j=1:9
    disp("seção"+j)
    load("pdt_01-08/05/avt/05setup0"+j+".mat")
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
    end
    end
end
end
Z24_d3=sens;
Z24_d3(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d3","Z24_d3")
disp("D3 - concluded")
%----------------------------------------------------------------
for j=1:9
    disp("seção"+j)
    load("pdt_01-08/06/avt/06setup0"+j+".mat")
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
    end
    end
end
end
Z24_d4=sens;
Z24_d4(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d4","Z24_d4")
disp("D4 - concluded")
%----------------------------------------------------------------
for j=1:9
    disp("seção"+j)
    load("pdt_01-08/08/avt/08setup0"+j+".mat")
for i=1:27
    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j-1,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j-1,:))
       sens(:,4*(j-1)+1)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j,:))
       sens(:,4*(j-1)+2)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+17,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+17,:))
       sens(:,4*(j-1)+3)=data(1:65000,i); 
    end
    end

    if strlength(strcat(labelshulp(i,:)))==strlength(strcat(name_sens(2*j+18,:)))
    if strcat(labelshulp(i,:))==strcat(name_sens(2*j+18,:))
       sens(:,4*(j-1)+4)=data(1:65000,i); 
    end
    end
end
end
Z24_d5=sens;
Z24_d5(:,[1,2,3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,35,36])=[]; %remove damaged sensors for reduce the final mesh to 7 sensors on Bern side (Alves, V., Cury, A. & Fortes, R. (2024))
save("Z24_d5","Z24_d5")
disp("D5 - concluded")
%----------------------------------------------------------------
%Final ordering of the 7 sensors:
% 105V, 110V, 115V, 120V, 125V, 130V, 135V.
