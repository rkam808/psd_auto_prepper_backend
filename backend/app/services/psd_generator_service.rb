require 'faraday'
require 'faraday/multipart'

class PsdGeneratorService
  PYTHON_SERVICE_URL = 'http://127.0.0.1:8000/process-model'

  def initialize(job_id)
    @job_id = job_id
  end

  def call
    job = ProcessingJob.find(@job_id)
    job.update(status: :processing)

    begin
      # 1. Get the binary data of the uploaded image
      # We download it into memory to send it to Python
      original_file_data = job.original_image.download
      filename = job.original_image.filename.to_s

      # 2. Prepare the connection to Python
      conn = Faraday.new do |f|
        f.request :multipart
        f.adapter :net_http
      end

      # 3. Send the POST request
      payload = {
        file: Faraday::Multipart::FilePart.new(
          StringIO.new(original_file_data),
          'image/png',
          filename
        )
      }

      response = conn.post(PYTHON_SERVICE_URL, payload)

      if response.success?
        # 4. Attach the result (The PSD) back to our Job
        job.prepped_psd.attach(
          io: StringIO.new(response.body),
          filename: 'prepped_model.psd',
          content_type: 'application/x-photoshop'
        )
        job.update(status: :completed)
      else
        puts "Python Service Failed: #{response.status} - #{response.body}"
        job.update(status: :failed)
      end

    rescue StandardError => e
      puts "Service Error: #{e.message}"
      puts e.backtrace
      job.update(status: :failed)
    end
  end
end
