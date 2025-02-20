clc; clear; close all;

% Read and convert the image
img = imread('boat.png'); % Change as needed
img = double(img);  % Convert to double

% Define JPEG Standard Luminance Quantization Table
Q = [16 11 10 16 24 40 51 61;
     12 12 14 19 26 58 60 55;
     14 13 16 24 40 57 69 56;
     14 17 22 29 51 87 80 62;
     18 22 37 56 68 109 103 77;
     24 35 55 64 81 104 113 92;
     49 64 78 87 103 121 120 101;
     72 92 95 98 112 100 103 99];

block_size = 8;
[M, N, C] = size(img);

% Ensure dimensions are multiples of block_size
M_adj = floor(M / block_size) * block_size;
N_adj = floor(N / block_size) * block_size;

% Precompute DCT and IDCT matrices
DCT_mat = dct_matrix(block_size);

% Initialize storage for compressed data
compressed_data = cell(C, M_adj / block_size, N_adj / block_size);

% JPEG ENCODER
for channel = 1:C
    prev_DC = 0; % Previous DC coefficient for DPCM
    for i = 1:block_size:M_adj
        for j = 1:block_size:N_adj
            % Extract 8x8 block
            block = img(i:i+block_size-1, j:j+block_size-1, channel);
            
            % Apply DCT
            dct_block = DCT_mat * block * DCT_mat';

            % Quantization
            quantized_block = round(dct_block ./ Q);

            % DPCM for DC coefficient
            DC_coeff = quantized_block(1,1);
            DPCM_value = DC_coeff - prev_DC;
            prev_DC = DC_coeff;

            % Zigzag Scanning
            AC_coeffs = zigzag_scan(quantized_block);
            AC_coeffs = AC_coeffs(2:end); % Remove DC component

            % Huffman Encoding (Placeholder)
            encoded_DC = huffman_encode(DPCM_value);
            encoded_AC = huffman_encode(AC_coeffs);

            % Store compressed data
            compressed_data{channel, (i-1)/block_size+1, (j-1)/block_size+1} = {encoded_DC, encoded_AC};
        end
    end
end

% JPEG DECODER (Reconstructing for Display)
reconstructed_img = zeros(M_adj, N_adj, C);

for channel = 1:C
    prev_DC = 0;
    for i = 1:block_size:M_adj
        for j = 1:block_size:N_adj
            % Retrieve compressed data
            encoded_DC = compressed_data{channel, (i-1)/block_size+1, (j-1)/block_size+1}{1};
            encoded_AC = compressed_data{channel, (i-1)/block_size+1, (j-1)/block_size+1}{2};

            % Huffman Decoding (Placeholder)
            DC_coeff = huffman_decode(encoded_DC) + prev_DC;
            prev_DC = DC_coeff;
            AC_coeffs = huffman_decode(encoded_AC);

            % Reconstruct block
            % Ensure AC_coeffs has exactly 63 elements (for 8x8 block)
            AC_coeffs = AC_coeffs(:); % Ensure it's a column vector
            missing_elements = 63 - length(AC_coeffs);
            
            if missing_elements > 0
                AC_coeffs = [AC_coeffs; zeros(missing_elements, 1)]; % Pad with zeros
            elseif missing_elements < 0
                AC_coeffs = AC_coeffs(1:63); % Trim excess values
            end
            
            % Concatenate DC and AC coefficients
            zz_vector = [DC_coeff; AC_coeffs];
            
            % Perform inverse zigzag scan
            quantized_block = izigzag_scan(zz_vector, block_size);

            dct_block = quantized_block .* Q;
            block = DCT_mat' * dct_block * DCT_mat;

            % Store reconstructed block
            reconstructed_img(i:i+block_size-1, j:j+block_size-1, channel) = block;
        end
    end
end

% Convert reconstructed image back to uint8
reconstructed_img = uint8(reconstructed_img);

% Save as an actual JPEG file
imwrite(reconstructed_img, 'compressed_output_boat.jpg');

% Display original and compressed file sizes
original_size = M * N * C;
compressed_size = dir('compressed_output_boat.jpg').bytes;
compression_ratio = original_size / compressed_size;

fprintf('Original Image Size: %d bytes\n', original_size);
fprintf('Compressed JPEG Size: %d bytes\n', compressed_size);
fprintf('Compression Ratio: %.2f\n', compression_ratio);

% Display original and compressed images
figure;
subplot(1,2,1); imshow(uint8(img)); title('Original Image');
subplot(1,2,2); imshow(imread('compressed_output_boat.jpg')); title('Compressed JPEG Image');

% --- Function: Compute DCT Matrix ---
function T = dct_matrix(N)
    T = zeros(N);
    for i = 0:N-1
        for j = 0:N-1
            if i == 0
                T(i+1, j+1) = sqrt(1/N);
            else
                T(i+1, j+1) = sqrt(2/N) * cos((pi * (2*j+1) * i) / (2 * N));
            end
        end
    end
end

% --- Zigzag Scan Function ---
function zz = zigzag_scan(block)
    zigzag_order = [
        1 2 6 7 15 16 28 29;
        3 5 8 14 17 27 30 43;
        4 9 13 18 26 31 42 44;
        10 12 19 25 32 41 45 54;
        11 20 24 33 40 46 53 55;
        21 23 34 39 47 52 56 61;
        22 35 38 48 51 57 60 62;
        36 37 49 50 58 59 63 64];
    zz = block(zigzag_order);
end

% --- Inverse Zigzag Scan Function ---
function block = izigzag_scan(zz, N)
    zigzag_order = [
        1 2 6 7 15 16 28 29;
        3 5 8 14 17 27 30 43;
        4 9 13 18 26 31 42 44;
        10 12 19 25 32 41 45 54;
        11 20 24 33 40 46 53 55;
        21 23 34 39 47 52 56 61;
        22 35 38 48 51 57 60 62;
        36 37 49 50 58 59 63 64];
    block = zeros(N);
    block(zigzag_order) = zz;
end

% --- Huffman Encoding Function (Placeholder) ---
function encoded = huffman_encode(data)
    % Placeholder for Huffman encoding
    encoded = data;
end

% --- Huffman Decoding Function (Placeholder) ---
function decoded = huffman_decode(encoded)
    % Placeholder for Huffman decoding
    decoded = encoded;
end
